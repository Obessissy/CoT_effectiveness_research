from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class CausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, dim_feedforward=config.ffn_dim,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True, layer_norm_eps=config.layer_norm_eps,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("causal_mask", torch.triu(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} > max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        padding_mask = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.layers(hidden, mask=self.causal_mask[:length, :length], src_key_padding_mask=padding_mask)
        return self.lm_head(self.final_norm(hidden))

    def answer_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

    def eos_loss(self, logits: torch.Tensor, input_ids: torch.Tensor, examples: list) -> torch.Tensor:
        """Optional protocol loss for EOS; disabled by default by the pilot contract."""
        rows = torch.arange(len(examples), device=logits.device)
        positions = torch.tensor([example.answer_end - 1 for example in examples], device=logits.device)
        targets = input_ids[rows, positions + 1]
        return nn.functional.cross_entropy(logits[rows, positions], targets)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.no_grad()
def generate(model: CausalTransformer, input_ids: torch.Tensor, eos_id: int, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    sequences = input_ids.clone()
    finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)
    for _ in range(max_new_tokens):
        logits = model(sequences[:, -model.config.max_seq_len:])[:, -1]
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        sequences = torch.cat([sequences, next_token[:, None]], dim=1)
        finished |= next_token.eq(eos_id)
        if bool(finished.all()):
            break
    return sequences, finished
