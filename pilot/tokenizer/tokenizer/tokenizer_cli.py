"""Train and audit the deterministic tokenizer for the answer-only pilot."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import platform
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


VOCAB_SIZE = 8192
SEED = 1234
CODEBOOK_SIZE = 32
BASE_SPECIAL_TOKENS = ("<BOS>", "<EOS>", "<PAD>", "<ANSWER>")
CODEBOOK_TOKENS = tuple(f"<Z_{index:03d}>" for index in range(CODEBOOK_SIZE))
SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + CODEBOOK_TOKENS
REQUIRED_FIELDS = ("question", "answer")
DEFAULT_VERSION_NAME = "v1_sp_bpe_8k_seed1234"


def project_root() -> Path:
    # The tokenizer now lives under pilot/tokenizer/tokenizer; the project root
    # is therefore five directory levels above this source file.
    return Path(__file__).resolve().parents[4]


def default_dataset_dir() -> Path:
    return project_root() / "mathematics_dataset" / "data" / "clean-100k"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / DEFAULT_VERSION_NAME


def require_sentencepiece():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "sentencepiece is required. Use the project Python environment and "
            "obtain approval before installing packages."
        ) from exc
    return spm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def validate_jsonl(path: Path) -> dict[str, Any]:
    count = 0
    answer_types: collections.Counter[str] = collections.Counter()
    for line_number, row in iter_jsonl(path):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {missing}")
        if not isinstance(row["question"], str) or not row["question"]:
            raise ValueError(f"{path}:{line_number}: question must be non-empty text")
        if not isinstance(row["answer"], str) or not row["answer"]:
            raise ValueError(f"{path}:{line_number}: answer must be non-empty text")
        answer_types[str(row.get("answer_type", "unknown"))] += 1
        count += 1
    return {"path": str(path), "count": count, "answer_types": dict(sorted(answer_types.items()))}


def serialize_example(question: str, answer: str, codebook_token: str | None = None) -> str:
    if codebook_token is not None and codebook_token not in CODEBOOK_TOKENS:
        raise ValueError(f"invalid codebook token: {codebook_token}")
    middle = f" {codebook_token}" if codebook_token is not None else ""
    return f"<BOS> {question}{middle} <ANSWER> {answer} <EOS>"


def build_training_corpus(train_path: Path, corpus_path: Path) -> int:
    count = 0
    with corpus_path.open("w", encoding="utf-8", newline="\n") as output:
        for _, row in iter_jsonl(train_path):
            output.write(serialize_example(row["question"], row["answer"]))
            output.write("\n")
            count += 1
    return count


def sentencepiece_options(corpus_path: Path, model_prefix: Path, vocab_size: int) -> dict[str, Any]:
    return {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix),
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "character_coverage": 1.0,
        "byte_fallback": True,
        "user_defined_symbols": list(SPECIAL_TOKENS),
        "unk_id": 0,
        "bos_id": -1,
        "eos_id": -1,
        "pad_id": -1,
        "hard_vocab_limit": True,
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "num_threads": 1,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": False,
        "add_dummy_prefix": False,
        "split_by_whitespace": True,
        "allow_whitespace_only_pieces": True,
        "max_sentence_length": 16384,
    }


def train_sentencepiece(corpus_path: Path, model_prefix: Path, vocab_size: int) -> None:
    spm = require_sentencepiece()
    # SentencePiece 0.2.0 on Windows cannot open paths containing non-ASCII
    # characters. Train in the system temp directory, then copy artifacts back.
    with tempfile.TemporaryDirectory(prefix="cot_tokenizer_train_") as temp:
        stage_dir = Path(temp)
        stage_corpus = stage_dir / "training_corpus.txt"
        stage_prefix = stage_dir / "tokenizer"
        shutil.copyfile(corpus_path, stage_corpus)
        spm.SentencePieceTrainer.train(
            **sentencepiece_options(stage_corpus, stage_prefix, vocab_size)
        )
        model_prefix.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage_prefix.with_suffix(".model"), model_prefix.with_suffix(".model"))
        shutil.copyfile(stage_prefix.with_suffix(".vocab"), model_prefix.with_suffix(".vocab"))


def load_processor(model_dir: Path):
    spm = require_sentencepiece()
    model_path = model_dir / "tokenizer.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"missing tokenizer model: {model_path}")
    return spm.SentencePieceProcessor(model_proto=model_path.read_bytes())


def percentile(sorted_values: Sequence[int], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_lengths(lengths: list[int]) -> dict[str, Any]:
    values = sorted(lengths)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def sequence_statistics(processor: Any, split_path: Path) -> dict[str, Any]:
    lengths = [
        len(processor.encode(serialize_example(row["question"], row["answer"]), out_type=int))
        for _, row in iter_jsonl(split_path)
    ]
    return summarize_lengths(lengths)


def sample_by_answer_type(path: Path, per_type: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: collections.Counter[str] = collections.Counter()
    for _, row in iter_jsonl(path):
        answer_type = str(row.get("answer_type", "unknown"))
        seen[answer_type] += 1
        bucket = selected[answer_type]
        if len(bucket) < per_type:
            bucket.append(row)
        else:
            candidate = rng.randrange(seen[answer_type])
            if candidate < per_type:
                bucket[candidate] = row
    return [row for answer_type in sorted(selected) for row in selected[answer_type]]


def normalized_for_audit(text: str) -> str:
    return " ".join(text.split())


def extract_decoded_answer(decoded: str) -> str:
    if "<ANSWER>" not in decoded or "<EOS>" not in decoded:
        raise ValueError(f"decoded serialization is missing markers: {decoded!r}")
    return decoded.split("<ANSWER>", 1)[1].rsplit("<EOS>", 1)[0].strip()


def build_sample_encodings(processor: Any, train_path: Path, output_path: Path, seed: int) -> dict[str, Any]:
    samples = sample_by_answer_type(train_path, per_type=2, seed=seed)
    recovered_exact = 0
    recovered_normalized = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in samples:
            serialized = serialize_example(row["question"], row["answer"])
            ids = processor.encode(serialized, out_type=int)
            pieces = processor.encode(serialized, out_type=str)
            decoded = processor.decode(ids)
            decoded_answer = extract_decoded_answer(decoded)
            exact = decoded_answer == row["answer"]
            normalized = normalized_for_audit(decoded_answer) == normalized_for_audit(row["answer"])
            recovered_exact += int(exact)
            recovered_normalized += int(normalized)
            audit = {
                "id": row.get("id"),
                "answer_type": row.get("answer_type"),
                "question": row["question"],
                "answer": row["answer"],
                "serialized": serialized,
                "token_ids": ids,
                "pieces": pieces,
                "decoded": decoded,
                "decoded_answer": decoded_answer,
                "answer_exact": exact,
                "answer_whitespace_normalized": normalized,
            }
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return {
        "sample_count": len(samples),
        "answer_types": sorted({str(row.get("answer_type")) for row in samples}),
        "recovered_exact": recovered_exact,
        "recovered_whitespace_normalized": recovered_normalized,
    }


def verify_special_tokens(processor: Any) -> dict[str, Any]:
    ids: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        encoded = processor.encode(token, out_type=int)
        if len(encoded) != 1:
            raise ValueError(f"special token is not atomic: {token} -> {encoded}")
        token_id = encoded[0]
        if processor.id_to_piece(token_id) != token:
            raise ValueError(f"special token round trip failed: {token} -> {token_id}")
        ids[token] = token_id
    if len(set(ids.values())) != len(ids):
        raise ValueError("special-token ID collision detected")
    return {
        "all_atomic": True,
        "all_ids_distinct": True,
        "ids": ids,
        "codebook_ids_contiguous": [ids[token] for token in CODEBOOK_TOKENS]
        == list(range(ids[CODEBOOK_TOKENS[0]], ids[CODEBOOK_TOKENS[0]] + CODEBOOK_SIZE)),
    }


def verify_byte_fallback(processor: Any) -> dict[str, Any]:
    probe = "π ∑ √ 中文 🙂"
    ids = processor.encode(probe, out_type=int)
    unk_id = processor.unk_id()
    return {
        "probe": probe,
        "token_ids": ids,
        "pieces": processor.encode(probe, out_type=str),
        "unk_id": unk_id,
        "contains_unk": unk_id in ids,
        "decoded": processor.decode(ids),
    }


def processor_signature(processor: Any) -> list[tuple[str, float]]:
    return [
        (processor.id_to_piece(index), processor.get_score(index))
        for index in range(processor.get_piece_size())
    ]


def reproducibility_check(
    corpus_path: Path,
    primary_model: Path,
    vocab_size: int,
    probe_texts: Iterable[str],
) -> dict[str, Any]:
    primary = load_processor(primary_model.parent)
    with tempfile.TemporaryDirectory(prefix="tokenizer_repro_", dir=str(primary_model.parent.parent)) as temp:
        temp_dir = Path(temp)
        copied_corpus = temp_dir / "training_corpus.txt"
        shutil.copyfile(corpus_path, copied_corpus)
        second_prefix = temp_dir / "tokenizer"
        train_sentencepiece(copied_corpus, second_prefix, vocab_size)
        second_model = second_prefix.with_suffix(".model")
        spm = require_sentencepiece()
        second = spm.SentencePieceProcessor(model_proto=second_model.read_bytes())
        signature_equal = processor_signature(primary) == processor_signature(second)
        encodings_equal = all(
            primary.encode(text, out_type=int) == second.encode(text, out_type=int)
            for text in probe_texts
        )
        return {
            "model_hash_equal": sha256_file(primary_model) == sha256_file(second_model),
            "vocab_hash_equal": sha256_file(primary_model.with_suffix(".vocab"))
            == sha256_file(second_prefix.with_suffix(".vocab")),
            "piece_and_score_signature_equal": signature_equal,
            "probe_encodings_equal": encodings_equal,
            "equivalent_result": signature_equal and encodings_equal,
            "note": (
                "Binary model hashes may differ because SentencePiece records model/input paths. "
                "Piece IDs, scores, vocab output, and probe encodings define equivalence."
            ),
        }


def manifest_split_count(dataset_manifest: dict[str, Any], split: str) -> int:
    try:
        return int(dataset_manifest["splits"][split]["kept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"manifest has no usable kept count for split {split!r}") from exc


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def ensure_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing tokenizer artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def run_training(dataset_dir: Path, output_dir: Path, vocab_size: int, seed: int) -> None:
    spm = require_sentencepiece()
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "valid.jsonl"
    dataset_manifest_path = dataset_dir / "manifest.json"
    for required in (train_path, valid_path, dataset_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    train_validation = validate_jsonl(train_path)
    valid_validation = validate_jsonl(valid_path)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    expected_train = manifest_split_count(dataset_manifest, "train")
    expected_valid = manifest_split_count(dataset_manifest, "valid")
    if train_validation["count"] != expected_train:
        raise ValueError(f"train count mismatch: file={train_validation['count']} manifest={expected_train}")
    if valid_validation["count"] != expected_valid:
        raise ValueError(f"valid count mismatch: file={valid_validation['count']} manifest={expected_valid}")

    ensure_new_output_dir(output_dir)
    corpus_path = output_dir / "training_corpus.txt"
    training_examples = build_training_corpus(train_path, corpus_path)
    model_prefix = output_dir / "tokenizer"
    train_sentencepiece(corpus_path, model_prefix, vocab_size)
    processor = load_processor(output_dir)

    special_report = verify_special_tokens(processor)
    byte_report = verify_byte_fallback(processor)
    sequence_report = {
        "serialization": "<BOS> question <ANSWER> answer <EOS>",
        "train": sequence_statistics(processor, train_path),
        "valid": sequence_statistics(processor, valid_path),
    }
    write_json(output_dir / "sequence_length_stats.json", sequence_report)

    sample_report = build_sample_encodings(processor, train_path, output_dir / "sample_encodings.jsonl", seed)
    if sample_report["recovered_whitespace_normalized"] != sample_report["sample_count"]:
        raise ValueError("one or more sampled answers were not recoverable after round trip")

    probes = [
        serialize_example("Solve 11*y - 14 = 19 for y.", "3"),
        serialize_example("Is 2 a factor of both 3 and 2?", "False", "<Z_000>"),
        "π ∑ √ 中文 🙂",
    ]
    reproducibility = reproducibility_check(corpus_path, output_dir / "tokenizer.model", vocab_size, probes)
    if not reproducibility["equivalent_result"]:
        raise ValueError("the second tokenizer training run was not equivalent")

    config = {
        "tokenizer_type": "sentencepiece_bpe",
        "sentencepiece_version": spm.__version__,
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": processor.get_piece_size(),
        "seed": seed,
        "normalization_rule_name": "identity",
        "byte_fallback": True,
        "serialization": "<BOS> {question} <ANSWER> {answer} <EOS>",
        "latent_serialization": "<BOS> {question} <Z_NNN> <ANSWER> {answer} <EOS>",
        "special_tokens": list(BASE_SPECIAL_TOKENS),
        "codebook_tokens": list(CODEBOOK_TOKENS),
        "special_token_ids": special_report["ids"],
        "active_codebook_examples": {
            "K=1": ["<Z_000>"],
            "K=16": list(CODEBOOK_TOKENS[:16]),
            "K=32": list(CODEBOOK_TOKENS),
        },
        "trainer_options": sentencepiece_options(corpus_path, model_prefix, vocab_size),
    }
    config["trainer_options"]["input"] = relative_to_project(corpus_path)
    config["trainer_options"]["model_prefix"] = relative_to_project(model_prefix)
    write_json(output_dir / "tokenizer_config.json", config)

    validation_report = {
        "all_checks_passed": True,
        "train_jsonl": train_validation,
        "valid_jsonl": valid_validation,
        "manifest_counts_match": True,
        "fit_input": relative_to_project(train_path),
        "validation_or_test_used_for_fit": False,
        "special_tokens": special_report,
        "byte_fallback": byte_report,
        "sample_round_trip": sample_report,
        "deterministic_reencode": all(
            processor.encode(probe, out_type=int) == processor.encode(probe, out_type=int)
            for probe in probes
        ),
        "reproducibility": reproducibility,
    }
    write_json(output_dir / "validation_report.json", validation_report)

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_dir": relative_to_project(dataset_dir),
        "input_split": "train.jsonl",
        "dataset_manifest_hash": sha256_file(dataset_manifest_path),
        "input_file_hash": sha256_file(train_path),
        "training_corpus_hash": sha256_file(corpus_path),
        "tokenizer_type": "sentencepiece_bpe",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": processor.get_piece_size(),
        "seed": seed,
        "serialization": "<BOS> question <ANSWER> answer <EOS>",
        "special_tokens": list(BASE_SPECIAL_TOKENS),
        "codebook_size_reserved": CODEBOOK_SIZE,
        "training_examples": training_examples,
        "validation_examples_for_statistics_only": valid_validation["count"],
        "fit_splits": ["train.jsonl"],
        "model_sha256": sha256_file(output_dir / "tokenizer.model"),
        "vocab_sha256": sha256_file(output_dir / "tokenizer.vocab"),
        "library": {"python": platform.python_version(), "sentencepiece": spm.__version__},
        "platform": platform.platform(),
        "reproducibility": reproducibility,
        "notes": (
            "All 32 codebook tokens are reserved as atomic pieces. They are not inserted "
            "into the answer-only fitting corpus. Experiments activate deterministic prefixes "
            "of the reserved set, such as Z_000..Z_015 for K=16."
        ),
    }
    write_json(output_dir / "tokenizer_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "manifest": manifest}, ensure_ascii=False, indent=2))


def command_encode(args: argparse.Namespace) -> None:
    processor = load_processor(args.model_dir.resolve())
    text = serialize_example(args.question, args.answer, args.codebook_token)
    payload = {
        "serialized": text,
        "token_ids": processor.encode(text, out_type=int),
        "pieces": processor.encode(text, out_type=str),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_decode(args: argparse.Namespace) -> None:
    processor = load_processor(args.model_dir.resolve())
    print(processor.decode(args.token_ids))


def command_stats(args: argparse.Namespace) -> None:
    processor = load_processor(args.model_dir.resolve())
    report = {
        split: sequence_statistics(processor, args.dataset_dir.resolve() / f"{split}.jsonl")
        for split in ("train", "valid")
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and audit a new tokenizer version")
    train_parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    train_parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    train_parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.set_defaults(handler=lambda args: run_training(
        args.dataset_dir, args.output_dir, args.vocab_size, args.seed
    ))

    encode_parser = subparsers.add_parser("encode", help="serialize and encode one example")
    encode_parser.add_argument("--model-dir", type=Path, default=default_output_dir())
    encode_parser.add_argument("--question", required=True)
    encode_parser.add_argument("--answer", required=True)
    encode_parser.add_argument("--codebook-token", choices=CODEBOOK_TOKENS)
    encode_parser.set_defaults(handler=command_encode)

    decode_parser = subparsers.add_parser("decode", help="decode token IDs")
    decode_parser.add_argument("--model-dir", type=Path, default=default_output_dir())
    decode_parser.add_argument("token_ids", nargs="+", type=int)
    decode_parser.set_defaults(handler=command_decode)

    stats_parser = subparsers.add_parser("stats", help="report train/valid sequence lengths")
    stats_parser.add_argument("--model-dir", type=Path, default=default_output_dir())
    stats_parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    stats_parser.set_defaults(handler=command_stats)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
