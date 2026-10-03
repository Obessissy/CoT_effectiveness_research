from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import sentencepiece as spm


class FrozenTokenizer:
    """Read-only adapter for the frozen pilot SentencePiece artifact."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir).resolve()
        model_path = self.model_dir / "tokenizer.model"
        config_path = self.model_dir / "tokenizer_config.json"
        manifest_path = self.model_dir / "tokenizer_manifest.json"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.processor = spm.SentencePieceProcessor(model_proto=model_path.read_bytes())
        self.config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        self.special_ids = dict(self.config.get("special_token_ids", {}))
        if not self.special_ids:
            for token in ("<BOS>", "<EOS>", "<PAD>", "<ANSWER>"):
                self.special_ids[token] = self.processor.piece_to_id(token)
            for i in range(32):
                token = f"<Z_{i:03d}>"
                self.special_ids[token] = self.processor.piece_to_id(token)
        self._validate()

    def _validate(self) -> None:
        for token, expected_id in self.special_ids.items():
            ids = self.processor.encode(token, out_type=int)
            if ids != [int(expected_id)]:
                raise ValueError(f"tokenizer contract violation for {token}: {ids} != {expected_id}")
        if len(set(self.special_ids.values())) != len(self.special_ids):
            raise ValueError("special-token IDs collide")

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode(list(ids))

    def piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(int(token_id))

    def id(self, token: str) -> int:
        value = self.processor.piece_to_id(token)
        if value < 0:
            raise KeyError(token)
        return int(value)

    def hash(self) -> str:
        return hashlib.sha256((self.model_dir / "tokenizer.model").read_bytes()).hexdigest()

    def codebook_token(self, index: int) -> str:
        if not 0 <= index < 32:
            raise ValueError("codebook index must be in [0, 31]")
        return f"<Z_{index:03d}>"
