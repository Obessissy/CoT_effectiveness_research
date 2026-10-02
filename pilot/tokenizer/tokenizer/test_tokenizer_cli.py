import unittest

import tokenizer_cli


class SerializationTests(unittest.TestCase):
    def test_answer_only_serialization(self):
        self.assertEqual(
            tokenizer_cli.serialize_example("1 + 1?", "2"),
            "<BOS> 1 + 1? <ANSWER> 2 <EOS>",
        )

    def test_latent_serialization(self):
        self.assertEqual(
            tokenizer_cli.serialize_example("1 + 1?", "2", "<Z_015>"),
            "<BOS> 1 + 1? <Z_015> <ANSWER> 2 <EOS>",
        )

    def test_invalid_codebook_token_is_rejected(self):
        with self.assertRaises(ValueError):
            tokenizer_cli.serialize_example("1 + 1?", "2", "<Z_032>")

    def test_reserved_codebook_is_contiguous(self):
        self.assertEqual(len(tokenizer_cli.CODEBOOK_TOKENS), 32)
        self.assertEqual(tokenizer_cli.CODEBOOK_TOKENS[0], "<Z_000>")
        self.assertEqual(tokenizer_cli.CODEBOOK_TOKENS[-1], "<Z_031>")


if __name__ == "__main__":
    unittest.main()
