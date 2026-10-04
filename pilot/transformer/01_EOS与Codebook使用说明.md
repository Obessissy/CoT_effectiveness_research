# Transformer Pilot：EOS、Codebook 与使用说明

这份说明解释 `pilot/transformer` 中的 EOS、codebook、oracle 和指标。当前阶段已有 EOS baseline 与 fixed-token 对照；最新实验决策请看 [04_Random实验决策与结果核验.md](04_Random实验决策与结果核验.md)，不要把本说明中的旧命令示例当成当前待办。

## 1. EOS 是什么？“加入”具体指什么？

一条样本的序列大致是：

```text
<BOS> 题目 <ANSWER> 答案 <EOS>
```

`<EOS>` 是“答案到这里结束”的标记。训练时有两件不同的事，容易混为一谈：

1. **EOS 是否参与训练 loss**：模型是否因没有在答案后预测 `<EOS>` 而受到惩罚。
2. **评测是否把 EOS 算作答案**：答案正确性和 answer likelihood 是否包含这个协议 token。

这两件事不应混淆。当前实现始终只把答案正文用于 `answer_nll` 和 answer log-likelihood；EOS 不算答案。默认训练也遵守 pilot 约定，把 EOS 从 loss 中 mask 掉。新增的 `--eos-loss-weight` 只影响训练，不会把 EOS 计入答案指标。

### 不训练 EOS（默认 `--eos-loss-weight 0`）

好处：严格保持原始 answer-only 目标，训练 loss 只代表答案 token；历史实验可以照此复现。

代价：模型没有被教过“答案后应该停”。自由生成可能一直重复、输出额外内容或达到 `--max-new-tokens` 仍未输出 EOS。此时 exact match、invalid 和 generation failure rate 会受停止行为影响，不能单独代表模型会不会做题。

### 训练 EOS（例如 `--eos-loss-weight 1.0`）

好处：模型会额外学到在答案后结束，生成评测更可用，failure rate 更有解释性。

代价：这改变了训练目标，是一个新的实验变体；训练日志里的总 loss 含有 EOS loss，不应直接和 EOS loss 为 0 的 answer-only 训练 loss 比大小。answer NLL 评测仍只计答案正文，因此可以单独观察答案条件概率。

代码把 EOS loss 作为独立项加入：

```text
总训练 loss = answer loss + eos_loss_weight × EOS loss
```

`eos_loss_weight=0` 是旧协议，`1.0` 是合理的首次试跑值，但这不是已经经过调参确定的最优权重。每个 run 的 manifest/checkpoint 会记录这个值。

### 对你现在最实用的建议

先用一个**新目录**跑 `--eos-loss-weight 1.0`，这样自由生成有停止监督。若要做严谨对照，再用相同 seed、epoch、数据和其他配置分别跑 EOS=0 与 EOS=1 的两个新 run。比较时分开看：

- `answer_nll`：只衡量给定正确答案前缀下，模型给答案正文的概率；
- `exact_match` / verifier：衡量生成出来的答案；
- `generation_failure_rate`：衡量是否在 token 上限前结束。

**不要拿训练 loss 数字直接比较 EOS 开/关的 run**，因为两种 run 的 loss 组成不同。

## 2. `codebook_mode` 是什么？

Codebook 是一组预留的特殊 token：`<Z_000>` 到 `<Z_031>`。启用时，它会插在题目和 `<ANSWER>` 之间：

```text
<BOS> 题目 <Z_017> <ANSWER> 答案 <EOS>
```

它不是自然语言解释，也不会自动产生 CoT。它只是一个可干预的条件 token。当前有三种模式：

| 模式 | 每条样本输入 | 用途 | 当前实现的注意点 |
|---|---|---|---|
| `none` | 不插入 Z token | direct answer 基线 | 推荐首先建立和评测的基准 |
| `fixed` | 所有样本插入同一个 Z token | 固定 token 对照，例如统一用 `<Z_000>` | 训练和评测必须使用相同 `--codebook-index` |
| `random` | 每条样本按 seed 和样本 id 确定性分配一个 Z token | 随机 token 对照；同一 seed/id 可复现 | 当前代码在 32 个 token 中分配；不是每个 epoch 重新随机 |

训练命令示例：

```powershell
# 不使用 codebook：直接答案基线
--codebook-mode none

# 所有训练题固定插入 Z_000
--codebook-mode fixed --codebook-index 0

# 每题按 seed/id 固定映射到 32 个 Z token 之一
--codebook-mode random
```

评测也需要使用和该 run 对应的条件。例如，固定 Z_000 的 run 应用：

```powershell
--codebook-mode fixed --codebook-index 0
```

如果训练时用 `none`，评测时却改成 `fixed`，就变成了“模型没按这个条件训练，却突然被塞入一个 token”；这不是有效的 fixed-token 对照。反过来也一样。

### 一个重要限制：当前 codebook run 会训练整个模型

目前训练代码把**整个 Transformer 的参数**交给 AdamW 更新；`fixed` 和 `random` 并不是“冻结主体，只训练 Z embedding”。因此即使训练固定 token，模型主体也会一起适配这个输入条件。这是当前代码行为，不等价于研究笔记中提到的“冻结主体后只训练 codebook”的实验设计。

因此接下来不要把当前 `fixed`/`random` run 描述成“仅学习了 latent token embedding”。若实验目标要严格冻结主体，需要先实现并验证参数冻结方案；否则要在实验记录里注明全模型参与训练，并确保各对照配置一致。

## 3. Oracle 与普通 codebook 模式不是一回事

`--oracle-k 1/16/32` 会用 gold answer 计算不同 Z token 下正确答案的 teacher-forcing likelihood，然后挑分数最高的 token。它在测试时看到了正确答案，所以是**答案泄漏下的 oracle 上界/诊断**，不是可部署方法。

使用前提：必须先有经过 codebook 条件训练的 checkpoint。对 `baseline_v1` 使用 oracle 不合适：其 manifest 记录的是 `codebook_mode=none`，模型没有在 Z 条件下被训练。它的 oracle 数字不能说明 codebook 无效或有效。

即便对训练过 codebook 的模型，oracle 提高 likelihood 也不代表生成准确率一定提高；两者是不同指标。selector 是以后才做的组件，它只能看题目、不能看 gold answer。当前项目还没有实现 selector。

## 4. 训练规模和“30M / 300M”

`baseline_v1` 的模型参数量是 29,479,936，约 **30M parameters**。约 354MB 指的是可恢复 checkpoint 文件，其中还包括 AdamW 优化器状态；不是模型有 300M 参数。

训练集每个 epoch 大约 2.86M serialized tokens。3 epoch 约 8.57M tokens，和 30M 参数相比训练量很小；这次 run 原本也是 pipeline pilot，不是质量结论。代码现在会把请求/实际处理 token 数和 tokens/parameter 写入新 run 的 `run_manifest.json`。

## 5. 历史训练与评测命令模板

按项目要求使用 `C:\ProgramData\anaconda3\python.exe`，并确保 Conda DLL 目录已在 `PATH` 中。如果当前 PowerShell 没有激活该环境，可先运行：

```powershell
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Library\bin;C:\ProgramData\anaconda3\Library\usr\bin;C:\ProgramData\anaconda3\Library\mingw-w64\bin;C:\ProgramData\anaconda3\Scripts;$env:PATH"
```

下面命令中的 `python` 就是指这个环境里的 Python。

### 第一步：建立 EOS 有监督的 direct baseline

从项目根目录运行。请保持路径和新 run 名称，**不要覆盖 `baseline_v1`**：

```powershell
python -m CoT_effectiveness_research.pilot.transformer.train `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --output-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_eos_v1 `
  --epochs 10 `
  --eos-loss-weight 1.0 `
  --skip-eval
```

10 epoch 是先观察训练趋势的起点，不是保证达到好准确率的承诺。训练结束后检查 `run_manifest.json` 中的 `training_complete`、`completed_epochs`、`data_stats` 和 `checkpoint_bytes`。

### 第二步：先做快速验证，再跑完整 validation

先用 100 条确认命令与输出格式：

```powershell
python -m CoT_effectiveness_research.pilot.transformer.evaluate `
  --checkpoint D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_eos_v1\checkpoints\last.pt `
  --tokenizer-dir D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234 `
  --dataset-dir D:\大四上\AIDR\Project\mathematics_dataset\data\clean-100k `
  --split valid --limit 100 `
  --output D:\大四上\AIDR\Project\CoT_effectiveness_research\pilot\transformer\runs\baseline_eos_v1\eval_valid_100.json
```

确认结果 JSON 里有合理的 `generation_failure_rate` 后，再去掉 `--limit 100` 跑完整 validation。之后再评估 `test_interpolate` 和 `test_extrapolate`。测试集用于最终报告，不要用它反复挑参数。

### 第三步：固定其他设置，只比较 no-token 与 fixed-token

建立两个**独立的新 run**：

1. `none`：作为 direct baseline；
2. `fixed --codebook-index 0`：所有训练例都插入 `<Z_000>`。

两者 seed、数据、epochs、batch size、EOS loss、学习率等应相同。评测时按训练模式加同样的参数。先只做 `K=1`（即一个固定 Z token），并确认 codebook run 的训练确实完成；此时先比较 likelihood、生成 exact match、generation failure，而不要立刻做 oracle。

### 第四步：再考虑 random 和 K=16/32

`random` 当前直接在 32 个 token 中按样本分配。如果要研究“只有 K=16 个 token 可用”，当前 CLI 没有 `active_k` 训练选项；不要把 `--oracle-k 16` 误认为训练了 K=16。oracle 的 K 只决定评测时枚举前多少个 token。

要开展 K=16/32 的严谨实验，下一步应先给训练数据的活动 codebook 大小增加显式配置，并把 `K` 写进 manifest，或分别用经过验证的 token 分配配置运行。待这一步确认后，再跑对应 oracle 分析。

## 6. 现在先记住这三句话

1. **EOS loss 开关影响模型有没有学会停，不会把 EOS 计入答案正确率或 answer NLL。**
2. **`codebook_mode` 必须训练和评测配套；none checkpoint 不能拿来证明 Z token 的效果。**
3. **先得到可信的 direct baseline，再做 fixed/random，再做 oracle，最后才是 selector。**

当前已有完整的 `baseline_eos_v1` 和 `baseline_fixed_z000_v1` validation。若继续实验，请参考 [04_Random实验决策与结果核验.md](04_Random实验决策与结果核验.md)，并用新目录保存；不要重复覆盖这些 run。
