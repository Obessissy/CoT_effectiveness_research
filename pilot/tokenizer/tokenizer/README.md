# Pilot Tokenizer

This directory contains the deterministic SentencePiece BPE tokenizer for the
answer-only pilot. The tokenizer reserves atomic pieces Z_000 through Z_031,
written with angle brackets in serialized examples. These pieces are not
inserted into the vocabulary-fitting corpus. Model experiments activate a
fixed prefix of the reserved set.

## Defaults

- SentencePiece BPE with an 8,192-piece vocabulary.
- Byte fallback enabled.
- Seed 1,234 recorded for reproducible sampling and experiment provenance.
- Only mathematics_dataset/data/clean-100k/train.jsonl fits the tokenizer.
- The answer-only layout is BOS, question, ANSWER, answer, EOS.
- The optional latent token is inserted immediately before ANSWER.
- Generated artifacts go to v1_sp_bpe_8k_seed1234.

## Windows environment

Use C:\ProgramData\anaconda3\python.exe as required by the parent instructions.
Before invoking it, prepend the Conda root, Library\bin, Library\usr\bin,
Library\mingw-w64\bin, and Scripts directories to PATH.

Run tokenizer_cli.py with the train command to create a new version. The
trainer refuses to overwrite a non-empty artifact directory. The encode,
decode, and stats commands provide the inspection API. Run Python unittest on
test_tokenizer_cli.py for the local unit tests.

The generated version records model and input hashes, full configuration,
train and validation sequence-length statistics, representative encodings for
all answer types, byte-fallback checks, and a second-run reproducibility audit.
