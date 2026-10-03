from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    pad_id: int = 3

    def parameter_count(self) -> int:
        # This mirrors CausalTransformer and is useful before constructing it.
        token = self.vocab_size * self.d_model
        position = self.max_seq_len * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model
            + 4 * self.d_model  # q/k/v/out biases
            + 2 * self.d_model * self.ffn_dim
            + self.ffn_dim + self.d_model
            + 4 * self.d_model  # two layer-norm weights and biases
        )
        final_norm = 2 * self.d_model  # final layer-norm weight and bias
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return token + position + self.n_layers * per_layer + final_norm + lm_head

    def to_dict(self) -> dict:
        result = asdict(self)
        result["parameter_count_estimate"] = self.parameter_count()
        return result


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1234
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    eos_loss_weight: float = 0.0
    epochs: int = 1
    max_steps: int | None = None
    eval_every_steps: int = 250
    log_every_steps: int = 25
    num_workers: int = 0
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
