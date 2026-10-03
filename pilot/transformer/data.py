from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import unicodedata
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import FrozenTokenizer


def normalize_answer_text(answer: str) -> str:
    """Conservative policy: NFC plus whitespace normalization only."""
    return " ".join(unicodedata.normalize("NFC", answer).split())


def serialize_example(question: str, answer_text: str, codebook_token: str | None = None) -> str:
    if codebook_token is not None and not (codebook_token.startswith("<Z_") and codebook_token.endswith(">")):
        raise ValueError(f"invalid codebook token: {codebook_token}")
    condition = f" {codebook_token}" if codebook_token else ""
    return f"<BOS> {question}{condition} <ANSWER> {answer_text} <EOS>"


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]
    answer_mask: list[bool]
    answer_start: int
    answer_end: int
    record: dict[str, Any]
    serialized: str


class SequenceTooLong(ValueError):
    pass


def encode_record(
    record: dict[str, Any], tokenizer: FrozenTokenizer, max_seq_len: int,
    codebook_token: str | None = None,
) -> EncodedExample:
    if not isinstance(record.get("question"), str) or not record["question"]:
        raise ValueError("record question must be non-empty text")
    if not isinstance(record.get("answer"), str) or not record["answer"]:
        raise ValueError("record answer must be non-empty text")
    answer_text = normalize_answer_text(record["answer"])
    serialized = serialize_example(record["question"], answer_text, codebook_token)
    ids = tokenizer.encode(serialized)
    answer_id = tokenizer.id("<ANSWER>")
    eos_id = tokenizer.id("<EOS>")
    try:
        marker = ids.index(answer_id)
        eos = ids.index(eos_id, marker + 1)
    except ValueError as exc:
        raise ValueError(f"serialized example has no answer/eos span: {serialized!r}") from exc
    answer_start, answer_end = marker + 1, eos
    if answer_start >= answer_end:
        raise ValueError("empty answer span after tokenization")
    if len(ids) > max_seq_len:
        raise SequenceTooLong(
            f"sequence length {len(ids)} exceeds max_seq_len={max_seq_len}; answer is never truncated"
        )
    labels = [-100] * len(ids)
    answer_mask = [False] * len(ids)
    # Logit at t predicts ids[t+1], so answer target ids [start:end] map to t [start-1:end-1].
    for target_index in range(answer_start, answer_end):
        labels[target_index - 1] = ids[target_index]
        answer_mask[target_index - 1] = True
    return EncodedExample(ids, labels, answer_mask, answer_start, answer_end, dict(record), serialized)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MathDataset(Dataset[EncodedExample]):
    def __init__(self, path: Path, tokenizer: FrozenTokenizer, max_seq_len: int,
                 codebook_mode: str = "none", codebook_index: int = 0, seed: int = 1234):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.codebook_mode = codebook_mode
        self.codebook_index = codebook_index
        self.seed = seed
        self.records = list(iter_jsonl(self.path))
        self.truncation_count = 0
        self._encoded: list[EncodedExample] = []
        for row in self.records:
            token = self._token_for(row)
            try:
                self._encoded.append(encode_record(row, tokenizer, max_seq_len, token))
            except SequenceTooLong:
                self.truncation_count += 1
                raise

    def _token_for(self, row: dict[str, Any]) -> str | None:
        if self.codebook_mode == "none":
            return None
        if self.codebook_mode == "fixed":
            return self.tokenizer.codebook_token(self.codebook_index)
        if self.codebook_mode == "random":
            digest = hashlib.sha256(f"{self.seed}:{row.get('id', '')}".encode()).digest()
            return self.tokenizer.codebook_token(digest[0] % 32)
        raise ValueError(f"unsupported codebook_mode={self.codebook_mode}")

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, index: int) -> EncodedExample:
        return self._encoded[index]


def collate_examples(examples: list[EncodedExample], pad_id: int) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(len(item.input_ids) for item in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    answer_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for row, item in enumerate(examples):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids)
        labels[row, :length] = torch.tensor(item.labels)
        answer_mask[row, :length] = torch.tensor(item.answer_mask)
        attention_mask[row, :length] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_mask": answer_mask,
        "attention_mask": attention_mask,
        "examples": examples,
    }


def make_loader(dataset: MathDataset, batch_size: int, seed: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=num_workers, collate_fn=lambda items: collate_examples(items, dataset.tokenizer.id("<PAD>")),
        pin_memory=torch.cuda.is_available(),
    )


def dataset_provenance(dataset_dir: Path, split: str) -> dict[str, Any]:
    path = Path(dataset_dir) / f"{split}.jsonl"
    return {"split": split, "path": path.as_posix(), "sha256": sha256_file(path), "examples": sum(1 for _ in iter_jsonl(path))}
