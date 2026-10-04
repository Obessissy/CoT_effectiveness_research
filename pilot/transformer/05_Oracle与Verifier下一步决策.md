# Oracle 与 Verifier：下一步决策

## 结论

当前最应该做的是 **完善 verifier 的高可靠覆盖**，而不是继续寻找“最佳 oracle token”。

oracle 诊断已经完成：它可以作为 gold-answer 上界记录下来，但不能作为可部署方法，也不能证明 CoT 或 codebook 有效。下一步应先让评测对更多答案类型给出可信的语义状态，再决定是否实现 question-only selector。

## 最新 random 结果

`baseline_random_v1` 与 direct baseline 使用相同的 10 epochs、seed=1234、EOS loss=1.0 和 validation 集：

| 条件 | answer NLL | exact match | verifier correct | unsupported | invalid | generation failure |
|---|---:|---:|---:|---:|---:|---:|
| random 普通生成 | 2.9857 | 10.68% (359/3360) | 346 | 890 | 182 | 0.06% |
| random oracle-k=32 | 不作为同一 NLL 比较 | 12.17% (409/3360) | 391 | 890 | 172 | 0.06% |

oracle 相对 random 普通生成提高约 1.49 个百分点，但它对每道题都用 gold answer 计算 32 个 token 的 teacher-forcing likelihood，再挑最高者。因此这是“看过答案后的上界”，不是模型在没有答案时能选择的 token。

oracle 选中的 token 分布也不支持“存在一个全局最佳 token”：32 个 token 都被选中过，出现最多的 `<Z_024>` 也只有 332/3360 条。实际部署必须学习 `q(z|question)`，这已经是 selector 问题，不是继续枚举 oracle 能解决的问题。

## 一个重要的评测文件注意点

本次生成了两个 oracle 文件：

- `runs/baseline_random_v1/eval_valid_full_oracle_k32.json`：命令没有显式传 `--codebook-mode random`，因此普通评测部分的输入是无 Z token；它的 oracle block 仍可读，但普通部分不适合与 random 训练条件直接比较。
- `runs/baseline_random_v1/eval_valid_full_random_oracle_k32.json`：显式传入 `--codebook-mode random`，普通评测和 oracle 都与 random 训练条件一致；后续引用应使用这个文件。

这两个文件的 oracle block 数值一致，是因为 oracle 函数本身会枚举 `<Z_000>`—`<Z_031>`；但普通评测部分的条件不同。

## 为什么 verifier 优先

当前 validation 的 3360 条样本中有 890 条 `unsupported`，约 26.5%，主要是：

| 类型 | 数量 |
|---|---:|
| expression | 546 |
| number_list | 226 |
| symbolic_list | 60 |
| time | 31 |
| base_number | 27 |

这不影响 exact match 和 answer NLL，但会限制语义准确率的解释。继续做 oracle 不会减少这些 unsupported，也不会让 gold-aware 指标变成可部署指标。

推荐按以下顺序完善 verifier：

1. `number_list`：严格解析分隔符、元素顺序和标量规则；这是覆盖量大且风险相对可控的一类。
2. `base_number`：显式读取题目/记录中的进制，禁止按普通十进制解析。
3. `time`：明确小时/分钟等格式和进位规则。
4. `symbolic_list`：先定义列表元素规则，再决定是否支持无序集合。
5. `expression`：最后考虑受限符号语法和等价性；没有安全规则时继续保持 `unsupported`，不要用未经审计的 `eval` 或宽松简化。

每一类都要加入正确、错误、格式错误和对抗性样例，并保持 `exact_match`、`verifier_status`、`unsupported`、`invalid` 分开报告。verifier 改进不能修改训练标签，也不能把旧结果重新解释成新规则下的历史结果而不标注 policy 版本。

## 推荐顺序

```text
random oracle-k=32（已完成：只作上界诊断）
        ↓
verifier：number_list / base_number / time + 单元测试
        ↓
用新 verifier policy 重新评测 none / fixed / random
        ↓
若仍无稳定差异：停止 codebook 扩展
        ↓
若要继续 latent condition：实现 question-only selector
```

不要再做“找一个全局最佳 Z token”：oracle 的 token 是逐题选择的，且依赖 gold answer。若要把 oracle 上界变成可用方法，必须训练只看题目的 selector，并在 selector 输入中排除 gold answer。
