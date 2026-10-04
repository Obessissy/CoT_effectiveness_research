# Random 实验决策与结果核验

## 先给结论

建议继续做 **一次** 严格配对的 random-token 对照，但不要马上重复多个随机种子、增加 epoch 或把它称为 CoT 增益实验。Random 的作用是回答一个较窄的问题：**在相同训练预算下，给每道题加入一个无语义、可复现的特殊 token，结果是否只是随机条件效应？**

如果 random 与 no-codebook、fixed token 都没有稳定改善，就可以把当前 codebook 分支作为“未观察到有效增益”的负结果，转向结构/训练目标或真正的 selector 设计，而不是继续堆 random 实验。

另一个概念边界：当前固定/随机 `<Z_xxx>` 实验没有自然语言 rationale，因此它测试的是 latent-token 条件，不是经典意义上“提供逐步 CoT 文本”的效果。如果研究问题明确是 CoT 文本的因果效果，后续需要单独设计同题配对的 rationale 条件与 answer-only 条件，不能把本实验标签直接称作 CoT。

## 重要的文件核验结果

本次检查的 `pilot/transformer/runs/` 中有：

- `baseline_eos_v1`：`codebook_mode=none`；
- `baseline_fixed_z000_v1`：`codebook_mode=fixed`、`codebook_index=0`；
- 没有发现名称或 manifest 明确对应 `z_001` 的 run、checkpoint 或评测 JSON。

因此，下面的数值是工作区中可以复核的 **z000 对照**，不是 z001：

| run | codebook | answer NLL | exact match | verifier correct | invalid | generation failure |
|---|---|---:|---:|---:|---:|---:|
| baseline_eos_v1 | none | 2.9859 | 10.86% (365/3360) | 355 | 183 | 0.03% |
| baseline_fixed_z000_v1 | fixed Z_000 | 2.9905 | 11.73% (394/3360) | 385 | 218 | 0.09% |

两者使用同一 validation 集、10 epochs、EOS loss=1.0、seed=1234 和相同模型配置。z000 的 exact match 高 0.86 个百分点，但 answer NLL 变差 0.0046，generation failure 也变差；配对 discordant exact-match 的双侧精确检验约 `p=0.075`，目前不足以称为稳定提升。这只是一个 paired comparison，不是 z001 结论。

## 如果你的 run 确实是 z_001

目前没有找到 z001 的 run 文件。如果它在另一位置，请按 manifest 核对 `codebook_index=1`，再与 no-codebook 逐样本配对，而不是只看两个百分比。至少检查：

1. `answer_nll`：越低越好；差异应与训练/评测噪声相比较。
2. `exact_match`：所有类型的字符串匹配，单独报告。
3. `verifier correct`：只在已支持的类型上解释；`unsupported` 不能算答错。
4. `invalid` 与 `generation_failure_rate`：防止“看似正确”其实是停止行为变化。
5. 每个 `answer_type`：尤其 integer、fraction、decimal、boolean，避免总体数字被题型比例掩盖。
6. 逐样本胜负数和 McNemar/二项配对检验：不要根据 0.5—1 个百分点的单次波动下结论。

若 z001 只在某一类提升、总体 NLL 不降且 invalid 增加，应记录为局部/不稳定变化，而不是“CoT 有效”。这里的 Z token 没有自然语言推理过程，所以 codebook 结果本身不能等同于 CoT 结果。

## Random 是否值得继续？

值得做一次，原因是它是项目要求的最小负对照；但它的解释边界很窄：

- `random` 当前按 `seed + sample id` 的稳定 hash，从 32 个 token 中分配 token；不是每个 epoch 重新抽样。
- 当前实现训练整个 Transformer，不是冻结主体、只训练 Z embedding。
- 因而 random 不是“随机 CoT”，也不是 selector；它测试的是随机条件 token 对全模型训练的影响。
- 只有在配置、seed、epoch、EOS loss、数据和评测协议都与 baseline/fixed 完全一致时，比较才有效。

推荐只运行一个 `baseline_random_v1`。如果它没有相对 direct baseline 的稳定改善，就停止扩展 random seeds；若它反而明显改善，先复核数据映射、checkpoint 和逐类型指标，再决定是否做 3-seed 重复。

## 推荐命令

从项目根目录执行，使用一个新的输出目录：

```powershell
python -m CoT_effectiveness_research.pilot.transformer.train `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --output-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_random_v1 `
  --epochs 10 `
  --eos-loss-weight 1.0 `
  --codebook-mode random `
  --skip-eval
```

评测时保持同一模式：

```powershell
python -m CoT_effectiveness_research.pilot.transformer.evaluate `
  --checkpoint D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_random_v1\checkpoints\last.pt `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --split valid `
  --output D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_random_v1\eval_valid_full.json `
  --codebook-mode random
```

不要给 random 运行 `--codebook-index`；该参数只用于 fixed 模式。

## Random 之后的停止条件

完成 direct、已确认的 fixed token（以及 z001 文件确认存在时）和 random 各一次完整 validation 后：

- 若 NLL、exact/verifier、invalid 和 failure 没有一致方向的改善：停止 codebook random 扩展，保留结果作为 negative control。
- 若只有 unsupported 类型变化：先完善并测试 verifier，不能据此声称模型能力提升。
- 若 integer 等已支持类型在多个指标、多个 split 上同步改善：再做一次独立 seed 复现；不要先增加 epoch。
- 若所有条件都低：优先检查 loss mask、学习率/优化器、数据题型混合和训练 token 预算；增加 random token 数量不是首选修复。

下一阶段若要研究真正的“可用 latent condition”，应实现 question-only selector，并确保 selector 输入不含 gold answer。oracle 只可作为 gold-answer 上界，不能代替部署方法。
