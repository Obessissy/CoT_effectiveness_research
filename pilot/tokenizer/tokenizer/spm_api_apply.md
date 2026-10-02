结论：`spm_api.py` 的核心调用方式是对的，尤其是用：

```
spm.SentencePieceProcessor(model_proto=p.read_bytes())
```

绕过 Windows 中文路径问题这一点是合理的。但目前脚本里对 `piece_to_id`、越界 ID 和 `get_score` 的理解需要修正。

注释 1 你关心的 `12345`、负数、分数、小数、`x^2`、多项式和括号，我已经实际测试过了。

## 1. 为什么 `sp.piece_to_id("1/3")` 是 0？

因为：

```
sp.piece_to_id("1/3")
```

查询的是：

> 词表里是否存在一个“完整且精确等于 `1/3` 的 piece”。

如果不存在，SentencePiece 会返回 `unk_id`。你的配置中：

```
unk_id = 0
```

所以结果是：

```
sp.piece_to_id("1/3") == 0
```

这不表示文本 `1/3` 被编码成了未知 token，也不表示分数无法处理。

真正的文本编码应该使用：

```
sp.encode("1/3", out_type=int)
sp.encode("1/3", out_type=str)
```

我实际测试得到：

```
文本：1/3
ids：    [8128, 421]
pieces：["1", "/3"]
decode：1/3
```

也就是说，`1/3` 被拆成两个 piece：

```
"1" + "/3"
```

而不是一个整体 piece。

如果输入前面有空格：

```
" 1/3"
```

结果是：

```
pieces：["▁1/3"]
```

你在 Windows 控制台中看到的 `�x1/3`，本质上应该是 `▁1/3` 中的 Unicode 空格标记 `▁` 显示乱码，不是 tokenizer 生成了奇怪的字符。

因此应当区分：

| 调用                  | 含义                                   |
| --------------------- | -------------------------------------- |
| `piece_to_id("1/3")`  | 查询词表中有没有一个完整的 `1/3` piece |
| `encode("1/3")`       | 把文本 `1/3` 按 tokenizer 实际规则切分 |
| `id_to_piece(421)`    | 查询 ID 421 对应的词表 piece           |
| `decode([8128, 421])` | 把 token ID 序列还原成文本             |

当前 `1/3` 的编码是正常的。

## 2. 我测试到的其他切分结果

### 普通分数

```
1/3
ids：    [8128, 421]
pieces：["1", "/3"]
-1/3
ids：    [2872, 421]
pieces：["-1", "/3"]
1 / 3
pieces：["1", "▁", "/", "▁3"]
```

这说明空格是否存在会影响切分。对于 DeepMind 数据，这通常不是问题，因为训练和推理序列会使用同样的序列化格式。但后续需要保持格式完全一致。

### 小数

```
1.5
pieces：["1.5"]
```

`1.5` 恰好是一个 piece。

```
-12.50
pieces：["-", "12.", "50"]
```

负号、整数部分和小数尾部被拆开了。

这不是错误。Tokenizer 不需要让每个数学对象都成为一个 token，重要的是：

- 编码可重复；
- decode 能恢复原文；
- 训练和测试使用相同 tokenizer；
- 数字外推时不要依赖具体数字词的记忆。

### 大整数

```
12345
pieces：["12", "345"]
```

`12345` 被拆成两个 piece。这个结果反而比较适合研究数字泛化，因为模型不会把每个完整数字都当作一个不可分解的词。

### 幂和多项式

```
x^2
pieces：["x", "<0x5E>", "2"]
```

其中：

```
<0x5E>
```

是 byte fallback 对 `^` 的表示。

更长的多项式：

```
x^2 + 3*x - 1
pieces：
["x", "<0x5E>", "2", "▁+", "▁3*", "x", "▁-", "▁1"]
```

这表示：

- `x` 是一个 piece；
- `^` 没有形成普通 learned piece，而是使用 byte fallback；
- `2` 是一个 piece；
- 带前导空格的 `+`、`-` 和数字可能与无空格版本不同。

### 括号和分式

```
(x+1)/(x-2)
pieces：
["(", "x", "+", "1)/(", "x", "-", "2)"]
```

这里出现了：

```
"1)/("
"2)"
```

这样的组合 piece。

这并不违反 tokenizer 标准。BPE 的目标是学习高频字符串片段，不是按照数学语法树切分。对于第一阶段 pilot，可以接受这种行为，只要序列能够正确还原。

不过它提示了一个后续研究风险：

> 如果未来重点研究数字组合、括号结构或数学运算外推，BPE 可能会把部分高频表达式片段合并得过于激进。

目前 pilot 只是验证完整流程，因此不建议立即改 tokenizer。先记录这个现象，之后可以把字符级或数字受控 tokenizer 作为单独消融。

## 3. 特殊 Token 的行为是正确的

例如：

```
<Z_017>
```

编码结果是：

```
ids：    [22]
pieces：["<Z_017>"]
```

并且配置中：

```
<Z_000> -> 5
<Z_001> -> 6
...
<Z_017> -> 22
```

32 个 codebook token 都是独立、连续、无语义的特殊 Token，这符合当前设计。

另外：

```
<BOS> 1/3 <ANSWER> 1/3 <EOS>
```

会被编码成：

```
["<BOS>", "▁1/3", "▁", "<ANSWER>", "▁1/3", "▁", "<EOS>"]
```

其中 `<BOS>`、`<ANSWER>`、`<EOS>` 是独立特殊 Token，分数部分依然按照普通文本规则切分。

## 4. `piece` 的 score 应该怎么理解？

你代码中的：

```
print(sp.get_score(5218))
```

实际得到：

```
piece：1/2
score：-4925.0
```

这个 `score` 不是：

- token 的概率；
- token 在当前句子中的概率；
- token 的 embedding；
- 模型预测该 token 的 log probability。

它是 SentencePiece 模型内部为 piece 保存的分数，用于 tokenizer 的分词模型或 piece 排序。不能直接把它解释成：

```
P("1/2")
```

也不能比较不同句子中这个 token 的生成概率。

另外，特殊 Token 的 score 是：

```
<BOS>   0.0
<Z_017> 0.0
```

这是正常的，因为它们是 user-defined special symbols，不是从语料统计学习出来的普通 BPE piece。

如果你想研究模型对答案的概率，应该使用后续 Transformer 的：

```
log p(answer | question, z)
```

而不是 tokenizer 的 `get_score()`。

## 5. `spm_api.py` 中有两个需要修正的理解问题

### 问题一：越界 ID 的注释不正确

当前代码有：

```
#sp.id_to_piece(99999)      # '<unk>'   越界返回 unk_id(0)
```

实际测试结果是：

```
sp.id_to_piece(99999)
IndexError: piece id is out of range
```

所以越界 ID 不会自动返回 `<unk>`。

只有真正的未知 token ID：

```
sp.unk_id() == 0
sp.id_to_piece(0) == "<unk>"
```

才对应 `<unk>`。

建议把注释改成：

```
# sp.id_to_piece(99999)  # raises IndexError: piece id is out of range
```

### 问题二：`piece_to_id()` 不适合用来检查一段文本

当前：

```
print(sp.piece_to_id('1/3'))
```

容易产生误解。

建议改成同时输出：

```
text = "1/3"
ids = sp.encode(text, out_type=int)
pieces = sp.encode(text, out_type=str)

print("text:", text)
print("ids:", ids)
print("pieces:", pieces)
print("decoded:", sp.decode(ids))
print("exact_piece_id:", sp.piece_to_id(text))
```

这样可以明确看到：

```
exact_piece_id: 0
```

只是说明没有完整 `1/3` piece，而不是说明编码失败。

## 6. `spm_api.py` 的路径还有一个问题

当前代码：

```
p = pathlib.Path("v1_sp_bpe_8k_seed1234/tokenizer.model")
```

依赖当前工作目录。

如果你在 tokenizer 目录运行，它可以工作；但如果你从项目根目录运行：

```
python CoT_effectiveness_research\pilot\tokenizer\tokenizer\spm_api.py
```

就可能找不到模型文件。

更稳妥的是使用脚本自身所在位置：

```
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "v1_sp_bpe_8k_seed1234"
MODEL_PATH = MODEL_DIR / "tokenizer.model"
sp = spm.SentencePieceProcessor(model_proto=MODEL_PATH.read_bytes())
```

这样无论从哪个目录启动，都能定位到同一个 tokenizer artifact。

## 7. 当前 tokenizer 是否符合 pilot 标准？

总体上符合。

已经满足：

- SentencePiece BPE；
- 8,192 词表；
- byte fallback；
- 只使用训练集拟合；
- 32 个特殊 codebook Token；
- 特殊 Token 是原子 piece；
- tokenizer 有 manifest、配置和验证报告；
- encode/decode 可以恢复数学文本；
- 多种答案类型有样例审计；
- 重复训练的 piece、score 和 probe encoding 一致。

当前不需要因为 `1/3` 不是一个 token 就重新训练 tokenizer。更重要的是：

1. `encode("1/3")` 能否稳定；
2. `decode(encode("1/3"))` 是否还原；
3. 训练和测试序列化格式是否一致；
4. 模型是否能在这种切分下学会答案；
5. 数字和表达式外推是否需要后续 tokenizer 对照。

当前建议只做两个小修正：

- 修正 `spm_api.py` 对 `piece_to_id` 和越界 ID 的注释/展示；
- 把模型路径改成相对于 `__file__` 的绝对解析路径。

暂时不建议改 tokenizer 的训练配置。