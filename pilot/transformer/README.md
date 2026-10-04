# Transformer pilot

中文实验文档已按编号整理。建议先读 [00_实验文档索引.md](00_实验文档索引.md)；关键解释文档依次为：

- [01_EOS与Codebook使用说明.md](01_EOS与Codebook使用说明.md)
- [02_结果解读与下一步指南.md](02_结果解读与下一步指南.md)
- [04_Random实验决策与结果核验.md](04_Random实验决策与结果核验.md)
- [05_Oracle与Verifier下一步决策.md](05_Oracle与Verifier下一步决策.md)
- [verifier/AGENTS.md](verifier/AGENTS.md)（verifier 专用开发约束）
- [03_手工运行记录.md](03_手工运行记录.md)（历史记录）

当前 workspace 中可核验的固定 token run 是 `baseline_fixed_z000_v1`，不是 z001。实验结果与是否继续 random 的最新判断请看 `04_Random实验决策与结果核验.md`。

This directory implements the first answer-only, decoder-only Transformer pipeline.
It uses the frozen SentencePiece artifact at
`../tokenizer/tokenizer/v1_sp_bpe_8k_seed1234` and never edits the tokenizer or
`mathematics_dataset/data/clean-100k`.

The default model is `d_model=512`, `n_layers=8`, `n_heads=8`, `ffn_dim=2048`,
`max_seq_len=128`, tied embeddings, and vocabulary size 8,192. Its measured
parameter count is 29,479,936 (about 30M parameters). The checkpoint is much
larger than that when AdamW state is included: the model weights are about
118MB and the two optimizer moment tensors add roughly 236MB, so a resumable
checkpoint around 350MB is expected. `run_manifest.json` records both values.
Only answer tokens are supervised by default; prompt, protocol delimiters, and
`<EOS>` are masked, as required by the current pilot contract. This has a
serious consequence: the model receives no stop-token target, so free
generation may not terminate at `<EOS>`. The optional `--eos-loss-weight 1.0`
adds a separately recorded protocol loss for EOS; it is a different run
variant, not a silent change to the answer-only baseline.

Run commands with `C:\ProgramData\anaconda3\python.exe` and the Conda DLL
directories on `PATH`, as required by the parent instructions:

```powershell
python -m CoT_effectiveness_research.pilot.transformer.train `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --output-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_v1
```

Training stops at the requested epoch count. It prints an `epoch_complete` and
`training_complete` record and saves a checkpoint after each completed epoch.
Full free-generation evaluation is intentionally a separate command, so it
cannot make an already-finished training run appear to be stuck. Add
`--eval-after-training` only when you explicitly want validation in the same
process. Resume with `--resume ...\checkpoints\last.pt`; completed epochs are
stored in the checkpoint and are not repeated.

## Evaluate a saved checkpoint

From the project root, this quick smoke evaluation scores 100 examples on the
validation split and writes a standalone JSON artifact:

```powershell
python -m CoT_effectiveness_research.pilot.transformer.evaluate `
  --checkpoint D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_v1\checkpoints\last.pt `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --split valid `
  --limit 100 `
  --output D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_v1\eval_valid_100.json
```

For full validation, remove `--limit 100`. To score both held-out test
distributions in one invocation, repeat `--split`; the output then contains a
`splits` object. (Previously repeating `--split` silently selected only the
last value.)

```powershell
python -m CoT_effectiveness_research.pilot.transformer.evaluate `
  --checkpoint D:\...\checkpoints\last.pt `
  --tokenizer-dir D:\...\v1_sp_bpe_8k_seed1234 `
  --dataset-dir D:\...\clean-100k `
  --split valid --split test_interpolate --split test_extrapolate `
  --limit 100 --output D:\...\eval_three_splits.json
```

Use a single `--split test_interpolate` or `--split test_extrapolate` when a
standalone artifact is preferable. The same
command can compare the direct baseline with one fixed special token by adding
`--codebook-mode fixed --codebook-index 0`, or with deterministic per-example
random tokens using `--codebook-mode random`. To enumerate an oracle over the
active codebook, add `--oracle-k 1`, `--oracle-k 16`, or `--oracle-k 32`; oracle
selection conditions on the gold answer and is not a deployable selector.

The evaluation JSON reports answer-token teacher-forcing likelihood, free
generation, exact normalized-string match, verifier status and per-answer-type
counts. Dataset hashes, tokenizer hash, configuration, answer policy version,
truncation counts, and checkpoint path are recorded in the run manifest.

The clean-100k train split contains about 2.86M serialized tokens per epoch.
Three epochs therefore expose this 30M-parameter model to only about 8.57M
tokens; that is enough for a pipeline smoke test but is a small training budget
for model quality. The manifest reports `tokens_per_parameter`; use a new run
directory and a larger `--epochs` value when doing a quality experiment (do not
overwrite an existing run).

`baseline_v1` was trained under the original answer-only contract, with EOS
masked, and has only 3 epochs. Its 2% exact match on a 100-example smoke sample
and 100% failure to emit EOS indicate that it is not a useful quality result.
Do not interpret the existing `--oracle-k` outputs as codebook evidence either:
the checkpoint manifest says `codebook_mode=none`, so the reserved Z embeddings
were never trained. Oracle enumeration on that checkpoint is a diagnostic over
untrained random embeddings, not the intended latent-token experiment.

`answer_policy.json` is provisional. Exact normalized-string metrics are always
reported; unsupported semantic types remain visible as `unsupported` until the
coordinator approves stronger rules.
