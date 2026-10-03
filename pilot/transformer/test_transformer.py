from __future__ import annotations

import unittest
from pathlib import Path

import torch

from .config import ModelConfig
from .data import encode_record, collate_examples
from .model import CausalTransformer
from .tokenizer import FrozenTokenizer
from .verifier import verify_answer


PILOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PILOT_DIR / "tokenizer" / "tokenizer" / "v1_sp_bpe_8k_seed1234"


class TransformerPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FrozenTokenizer(TOKENIZER_DIR)

    def test_special_tokens_are_atomic_and_stable(self):
        self.assertEqual(self.tokenizer.id("<Z_017>"), 22)
        self.assertEqual(self.tokenizer.encode("<Z_017>"), [22])

    def test_answer_loss_mask_is_shifted_and_excludes_prompt_and_eos(self):
        row = {"id": "a", "question": "What is 2+1?", "answer": "3", "answer_type": "integer", "split": "train"}
        example = encode_record(row, self.tokenizer, 128)
        marker = self.tokenizer.id("<ANSWER>")
        eos = self.tokenizer.id("<EOS>")
        self.assertEqual(example.input_ids[example.answer_start - 1], marker)
        self.assertEqual(example.input_ids[example.answer_end], eos)
        self.assertTrue(all(label == -100 for label in example.labels[:example.answer_start - 1]))
        self.assertEqual(example.labels[example.answer_start - 1], example.input_ids[example.answer_start])
        self.assertEqual(example.labels[example.answer_end - 1], -100)
        self.assertEqual(example.labels[example.answer_end], -100)

    def test_batch_forward_and_backward(self):
        records = [
            {"id": "a", "question": "1+2?", "answer": "3", "answer_type": "integer"},
            {"id": "b", "question": "2+2?", "answer": "4", "answer_type": "integer"},
        ]
        examples = [encode_record(row, self.tokenizer, 128) for row in records]
        batch = collate_examples(examples, self.tokenizer.id("<PAD>"))
        config = ModelConfig(vocab_size=self.tokenizer.vocab_size, d_model=32, n_layers=2, n_heads=4, ffn_dim=64, max_seq_len=128)
        model = CausalTransformer(config)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = model.answer_loss(logits, batch["labels"])
        self.assertEqual(tuple(logits.shape[:2]), tuple(batch["input_ids"].shape))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_verifier_conservative_rules(self):
        self.assertEqual(verify_answer({"answer_type": "fraction", "answer": "1/2"}, "2/4")["status"], "correct")
        self.assertEqual(verify_answer({"answer_type": "integer", "answer": "3"}, "3 trailing")["status"], "invalid")
        self.assertEqual(verify_answer({"answer_type": "expression", "answer": "x"}, "x")["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
