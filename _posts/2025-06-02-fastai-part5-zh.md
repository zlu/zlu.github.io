---
layout: post
title: "使用 fastai 进行文本分类的简明指南 - Fastai Part 5"
date: 2025-05-30
comments: true
tags:
  - python
  - artificial intelligence
  - machine learning
  - fastai
  - PyTorch
description: "使用 fastai 进行文本分类的简明指南 - Fastai Part 5"
---

## 子词分词
对于中文而言，由于不存在"词"的概念，fastai通过`SentencePiece`提供的`subword`支持变得至关重要。由于词与词之间没有空格：
```
我喜欢学习 （I like studying)
```
SentencePiece会将其分割为：
```
我/喜欢/学习/
```
这是通过直接在原始中文文本上进行无监督学习过程来完成的，使模型能够泛化到新词和新表达。这对中文很重要，因为它允许模型学习正确的词边界，避免以意想不到的方式分割词语。因此，我们可以将它们应用于fastai的NLP管道中使用的AWD-LSTM或基于Transformer的模型。

下面是另一个例子：
```
我喜欢吃辣椒。
▁我 ▁喜欢 ▁吃 ▁辣 ▁椒 ▁。
```
每个token要么是一个字符，要么是一个频繁出现的组合，这是模型从训练数据（原始文本）中随时间学习到的。`_`标记了新子词的开始。如你所见，它正确地将`喜`（快乐）和`欢`（欢喜）组合为单个token`喜欢`（喜欢）。

具体来说，子词分词可以这样完成：

```python
# 假设 `text` 是原始文本
def subword(vocab_size):
  sw = SubwordTokenizer(vocab_size=vocab_size)
  sw.setup(text)
  return ' '.join(first(sw([text])))
```

注意，根据词汇表的大小，子词分词可能会产生不同的结果。词汇表越大，每句话的token越少，训练时间越快，但嵌入矩阵也越大。这就是为什么我们需要找到平衡。

## 数值化 - 将Token转换为数字

在本系列的第2部分中，我们学习了如何将图像转换为数字以进行分类任务。同样的原理适用于文本。计算机只能处理数字，所以我们需要将前一步的token转换为数字。然后我们可以将它们输入到神经网络中。

我们将利用fastai的`Numericalize()`将token转换为整数。这是通过创建一个`Vocab`对象来完成的，它是token到整数的映射。`Numericalize()`然后使用这个映射将token转换为整数。然后我们可以将它们输入到fastai的`Datasets`对象中，该对象对整个数据集应用相同的转换。结果的`dataset.items`将包含整数。

下面是一个示例：

```python
from fastai.text.all import *
from fastcore.basics import noop

tokens = [["I", "love", "deep", "learning"], ["Fastai", "makes", "it", "simple"]]

# 应用 noop + numericalize
dsets = Datasets(tokens, [[noop, Numericalize()]])

# 显示词汇表
vocab = dsets[0][0].vocab
print("Vocab:\n", vocab)

# 显示数值化后的数据
for i, item in enumerate(dsets):
    print(f"Sentence {i+1}: {item[0]}")
```

让我们可视化正在发生的事情：

```python
分词文本 → [ "Fastai", "makes", "it", "simple" ]
                   ↓
               Numericalize
                   ↓
整数IDs    → [ 6, 7, 8, 9 ]

其中：
    vocab = { "Fastai": 6, "makes": 7, "it": 8, "simple": 9, ... }
```

## DataLoader创建

我们在本系列的前几部分中已经看到了`DataLoader`。它接收原始或处理过的数据（如前一步骤的数值化文本）并将其转换为批次。这对于训练神经网络很重要，因为它允许模型以批次方式查看数据，这更高效和稳定。这里有两个重要概念需要理解：

1. 批处理

神经网络使用**批次**数据训练更快更可靠。所以前一步的输出：
```python
[2, 3, 4, 5]  # "I love deep learning"
[6, 7, 8, 9]  # "Fastai makes it simple"
```
被组合成一个批次。

2. 填充

由于序列数据结构通常具有可变长度，而张量**必须**具有相同大小（矩形）才能适合GPU内存，我们需要将序列填充到相同长度。例如，两个可变序列：
```
原始：           [2, 3, 4, 5]
                [6, 7]
填充后：         [2, 3, 4, 5]
                [6, 7, 0, 0]
```                  
将被填充到相同长度。

假设我们之前已经创建了一个`TextDataLoaders`，如下所示：

```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
```

然后我们可以这样创建一个填充批次：

```python
x, y = dls.one_batch()
print(type(x))  # torch.Tensor
print(x.shape)  # 例如，torch.Size([64, 72]) — 64个样本，每个72个token长
```

## 语言模型微调

正如我们在本系列前一部分所做的那样，我们可以这样微调语言模型：

```python
# 创建语言模型学习器
learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# 在你的语料库上微调
learn.fine_tune(1)
```

`fine_tune(1)`在学习对象上运行一个epoch的训练。这里预训练的基础模型被`冻结`，意味着在训练期间其权重不会更新。这很重要，因为我们只想专注于训练`头部`（或新添加的分类层）。这允许模型调整其最终层以适应新任务，而不会破坏预训练的表示。然后整个模型被`解冻`（包括基础模型），_所有_层一起微调。当我们传递`1`给`fine_tune`时，fastai将运行1个epoch，基础模型保持冻结，但跳过解冻步骤。所以我们需要传递一个大于1的数字来微调整个模型。

## 文本分类

最后，执行推理：
```python
learn.predict("I love deep learning")
```

这完成了我们应用RNN进行文本分类的旅程。在本系列的下一部分中，我们将深入探讨如何从头构建RNN网络！ 