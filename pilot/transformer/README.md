# Transformer pilot

中文实验文档已按编号整理。建议先读 [00_实验文档索引.md](00_实验文档索引.md)；关键解释文档依次为：

- [01_EOS与Codebook使用说明.md](01_EOS与Codebook使用说明.md)
- [02_结果解读与下一步指南.md](02_结果解读与下一步指南.md)
- [04_Random实验决策与结果核验.md](04_Random实验决策与结果核验.md)
- [05_Oracle与Verifier下一步决策.md](05_Oracle与Verifier下一步决策.md)
- [verifier/AGENTS.md](verifier/AGENTS.md)（verifier 专用开发约束）
- [06_Verifier_v3抽样审计.md](06_Verifier_v3抽样审计.md)（新版 verifier 抽样结果）
- [07_阶段性综合评估与组会汇报.md](07_阶段性综合评估与组会汇报.md)（当前结果总评与下一步建议）
- [08_三条件Checkpoint验证评测.md](08_三条件Checkpoint验证评测.md)（最新三个条件的 best/last 验证结果）
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

After every completed epoch, training now performs a fast teacher-forced
validation loss on a fixed, reproducible stratified sample: 100 `integer`, 100
`decimal`, and 100 `fraction` examples by default. The sample is selected once
from the validation JSONL using the run seed and reused at every epoch; it is
not simply the first 100 rows. This gives a stable loss curve without paying
the cost of free generation. The records are printed as an `epoch_metrics` JSON
event and written to `loss_history.json`; the selected IDs and sampling policy
are also stored in `run_manifest.json`. Change the sample size with
`--epoch-valid-per-type N`, or disable this curve with
`--no-epoch-validation`. Use the separate `evaluate.py` command for final
generation/exact-match/verifier metrics.

When epoch validation is enabled, the checkpoint with the lowest sampled
validation answer NLL is additionally saved as `checkpoints/best.pt`;
`run_manifest.json` records `best_epoch` and its metric. `checkpoints/last.pt`
remains the final epoch and may be overfit, so quality evaluation should
normally use `best.pt` after inspecting the curve.

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

## 20 epoch follow-up

Replacing the existing 10 epoch runs with fresh 20 epoch runs is reasonable as
an optimization diagnostic, but it should not overwrite the old directories.
Use new names such as `baseline_eos_v2_20ep`, `baseline_fixed_z000_v2_20ep`,
and `baseline_random_v2_20ep`. Inspect `loss_history.json` first: if both train
and sampled validation answer NLL continue decreasing at epoch 10, 20 epochs
tests whether the pilot was simply under-trained; if train NLL decreases while
validation NLL plateaus or rises, more epochs are not the right fix. Keep the
same seed, EOS weight, batch size, learning rate, data and verifier policy for
the three runs. Because adding a Z token changes sequence length, record the
processed-token count and do not call the comparison strictly compute-matched.
