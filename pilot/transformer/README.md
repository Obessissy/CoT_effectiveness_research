# Transformer pilot

This directory implements the first answer-only, decoder-only Transformer pipeline.
It uses the frozen SentencePiece artifact at
`../tokenizer/tokenizer/v1_sp_bpe_8k_seed1234` and never edits the tokenizer or
`mathematics_dataset/data/clean-100k`.

The default model is `d_model=512`, `n_layers=8`, `n_heads=8`, `ffn_dim=2048`,
`max_seq_len=128`, tied embeddings, and vocabulary size 8,192. Its measured
parameter count is 29,479,936. Only targets in the answer span are supervised;
the prompt, `<ANSWER>`, and `<EOS>` are masked.

Run commands with `C:\ProgramData\anaconda3\python.exe` and the Conda DLL
directories on `PATH`, as required by the parent instructions:

```powershell
python -m CoT_effectiveness_research.pilot.transformer.train `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --output-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_v1
```

Resume with `--resume ...\checkpoints\last.pt`. Evaluate any saved checkpoint
on `valid`, `test_interpolate`, or `test_extrapolate` using `evaluate.py`.
`--oracle-k 1|16|32` writes a separate gold-answer-conditioned oracle analysis;
it is explicitly not a deployable selector. Dataset hashes, tokenizer hash,
configuration, answer policy version, truncation counts, and checkpoint path
are written to each run manifest.

`answer_policy.json` is provisional. Exact normalized-string metrics are always
reported; unsupported semantic types remain visible as `unsupported` until the
coordinator approves stronger rules.
