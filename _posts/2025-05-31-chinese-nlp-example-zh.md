---
layout: post
title: "中文NLP with fastai - Fastai Part 4"
date: 2025-05-30
comments: true
categories: 
  - machine learning, NLP
tags:
  - python
  - artificial intelligence
  - machine learning
  - fastai
  - PyTorch
description: "中文NLP with fastai - Fastai Part 4"
---
## 使用fastai进行自然语言处理

在之前的教程中，我们已经了解了如何利用预训练模型并对其进行微调，以执行图像分类任务（MNIST）。应用于图像的迁移学习原理同样也可以应用于NLP任务。在本教程中，我们将使用名为AWD_LSTM的预训练模型来对中文电影评论进行分类。AWD_LSTM是LSTM的一种变体。LSTM是一种循环神经网络（RNN），专为处理长文本序列而设计。我们将在后续教程中详细讨论RNN。

## 使用fastai进行中文自然语言处理：实用示例

中文语言处理是一项具有挑战性的任务，因为大多数NLP模型都是用英语等西方语言训练的。与英语不同，中文不使用空格来分隔单词，这使得分词更具挑战性。幸运的是，有像jieba这样的库可以进行中文分词。Jieba和pkuseg是两个专为有效处理中文分词而设计的库。只要是在中文语料库上训练的，可以使用预训练的词嵌入，如Word2Vec、Glove或FastText。在本指南的最后，我将向您展示如何使用Google的BERT变体——中文BERT来捕捉中文文本中的上下文。XLM-RoBERTa是另一个在中文文本上表现良好的多语言模型。除了中文BERT外，还有许多本土模型，如百度的ERNIE（Enhanced Representation through kNowledge IntEgration）和PaddleNLP，阿里巴巴的FastBERT和AliceMind，以及腾讯的TecentPretrain和Chinese Word Vectors。

## 处理流程
广义上讲，NLP任务有两个基本模块：文本预处理和文本分类。

在文本预处理中，我们希望以计算机能够解释的方式准备文本。即使对于RNN来说，解释文本的上下文含义也是一项非常复杂的任务。Transformer和自注意力机制在这一领域取得了突破（因此在最后有transformer的例子）。为了简单起见，我们现在将主要关注**分词**和**词嵌入**步骤。

### 分词

分词是将文本转换为"标记"的操作，这些标记可以是字符（"a"、"b"、"c"，...）或单词（"hello"、"world"，...），甚至是子字符串，取决于模型的粒度。这就是中文语言变得有趣的地方，因为与英语或基于字母的语言不同，即使是中文字符（我，喜，欢，爱，中）本身也带有含义！因此，中文的词分割变成了一项更艰巨的任务，因为与英语不同，英语中的单词是由空格分隔的，中国人必须通过阅读和记忆来学习如何识别词的边界。因此，需要特殊的算法来分割中文文本。此外，中文文本中的外来词、数字和符号需要特殊处理。

### 词嵌入
词嵌入是一种将单词表示为向量的方法。在上一个教程中，我们看到了如何将MNIST数据集（灰度图像）转换为3D向量（高度、宽度、颜色）。我们在这里做的概念上也很相似。这些向量的特殊之处在于它们是从大量文本中学习的，那些含义相似的词在高维向量空间中彼此接近（物以类聚）。在本教程中，我们将创建一个自定义的fastai `DataBlock`，`ChineseTextBlock`，用于分词和嵌入中文文本。

### 文本分类
文本分类是为一段文本分配标签的任务。例如，我们可以将电影评论分类为正面或负面。我们将使用fastai的数据加载器和`AWD_LSTM`来构建一个文本分类器。

## 设置和导入


```python
# 如果需要，安装所需的包
# !pip install fastai jieba
```


```python
from fastai.text.all import *
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 中文文本示例

为了演示目的，我们将创建一个小型中文电影评论数据集。在实际应用中，您将加载自己的数据集。


```python
# 中文电影评论示例（正面和负面）
positive_reviews = [
    "这部电影非常精彩，演员的表演令人印象深刻。",
    "剧情紧凑，特效惊人，是今年最好看的电影之一。",
    "导演的手法很独特，将故事讲述得引人入胜。",
    "音乐配乐恰到好处，为电影增添了不少气氛。",
    "这是一部让人回味无穷的佳作，值得一看。"
]

negative_reviews = [
    "情节拖沓，演员表演生硬，浪费了我的时间。",
    "特效做得很差，剧情漏洞百出，非常失望。",
    "导演似乎不知道自己想要表达什么，整部电影混乱不堪。",
    "对白尴尬，角色塑造单薄，完全不推荐。",
    "这部电影毫无亮点，是我今年看过最差的一部。"
]

# 创建数据框
reviews = positive_reviews + negative_reviews
labels = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)

df = pd.DataFrame({'text': reviews, 'label': labels})
df = df.sample(frac=1).reset_index(drop=True)  # 打乱数据

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>这是一部让人回味无穷的佳作，值得一看。</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>剧情紧凑，特效惊人，是今年最好看的电影之一。</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>导演的手法很独特，将故事讲述得引人入胜。</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>特效做得很差，剧情漏洞百出，非常失望。</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>音乐配乐恰到好处，为电影增添了不少气氛。</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



## 中文文本分词

让我们探索中文文本的不同分词方法。

### 1. 使用Jieba进行词级分词

我们需要区分在中文NLP上下文中"词"的含义。中文词由提供含义的汉字组成。例如，词"中国"由两个汉字"中"和"国"组成。词"中国"与"中国是一个伟大的国家"中的"中国"含义不同（中国是一个伟大的国家）。后一句中的"中国"是一个名词短语，而前一句中的"中国"是一个名词。前一句中的"中国"是一个单词，而后一句中的"中国"是两个词。前一句中的"中国"是一个单独的标记，而后一句中的"中国"是两个标记。
然而在英语中，一个词就是一个词，比如"China"。所以中文NLP中的词在概念上更类似于英语中的"子词"。


```python
def chinese_word_tokenizer(text):
    """使用Jieba进行中文词分割"""
    # 处理Path对象，通过读取文件
    if hasattr(text, 'read_text'):
        text = text.read_text(encoding='utf-8')
    elif hasattr(text, 'open'):
        text = text.open(encoding='utf-8').read()
    
    # 如果有混合的英文文本，转换为小写
    text = str(text).lower()
    # 使用Jieba分割词
    words = jieba.cut(text)
    return list(words)

# 示例
sample_text = "这部电影非常精彩，演员的表演令人印象深刻。"
word_tokens = chinese_word_tokenizer(sample_text)
print(f"词级标记: {word_tokens}")
print(f"标记数量: {len(word_tokens)}")
```

    词级标记: ['这部', '电影', '非常', '精彩', '，', '演员', '的', '表演', '令人', '印象', '深刻', '。']
    标记数量: 12


### 2. 字符级分词


```python
def chinese_char_tokenizer(text):
    """在字符级别对中文文本进行分词"""
    # 处理Path对象，通过读取文件
    if hasattr(text, 'read_text'):
        text = text.read_text(encoding='utf-8')
    elif hasattr(text, 'open'):
        text = text.open(encoding='utf-8').read()
    
    # 如果不是字符串，转换为字符串
    text = str(text)
    # 如果有空格，移除空格
    text = text.replace(" ", "")
    # 分割成字符
    return list(text)

# 示例
char_tokens = chinese_char_tokenizer(sample_text)
print(f"字符级标记: {char_tokens}")
print(f"标记数量: {len(char_tokens)}")
```

    字符级标记: ['这', '部', '电', '影', '非', '常', '精', '彩', '，', '演', '员', '的', '表', '演', '令', '人', '印', '象', '深', '刻', '。']
    标记数量: 21


### 3. 为中文定制的fastai分词器

在fastai的NLP框架中，特殊标记在帮助模型理解文本结构方面起着至关重要的作用。在下面的代码片段中，您将看到`xxbos`，它告诉模型一个新句子正在开始。其他常用的特殊标记包括`xxmaj`（用于大写），`xxup`（用于全大写），`xxrep`（用于重复一个词），和`xxwrep`（用于重复一个带空格的词）。



```python
# 定义一个可以截断显示的字符串类
class TitledStr(str):
    """一个可以为显示目的而截断的字符串"""
    def truncate(self, n):
        return TitledStr(self[:n] + '...' if len(self) > n else self)
        
    def show(self, ctx=None, **kwargs):
        "在上下文中显示文本"
        return show_text(self, ctx=ctx, **kwargs)
        
def show_text(text, ctx=None, **kwargs):
    "显示文本的辅助函数"
    if ctx is None: ctx = {'text': text}
    else: ctx['text'] = text
    return ctx

class ChineseTokenizer(Transform):
    def __init__(self, tokenizer_func=chinese_word_tokenizer):
        self.tokenizer_func = tokenizer_func
        
    def encodes(self, x):
        tokens = self.tokenizer_func(x)
        # 添加特殊标记，如BOS（句子开始）
        tokens = ['xxbos'] + tokens
        return tokens
    
    def decodes(self, x):
        text = ''.join(x) if isinstance(x[0], str) and len(x[0]) == 1 else ' '.join(x)
        # 创建一个带有截断方法的文本对象
        return TitledStr(text)

# 为两种分词方法创建实例
word_tokenizer = ChineseTokenizer(chinese_word_tokenizer)
char_tokenizer = ChineseTokenizer(chinese_char_tokenizer)

# 示例
print("词级分词器:")
print(word_tokenizer.encodes(sample_text))
print("\n字符级分词器:")
print(char_tokenizer.encodes(sample_text))
```

    词级分词器:
    ['xxbos', '这部', '电影', '非常', '精彩', '，', '演员', '的', '表演', '令人', '印象', '深刻', '。']
    
    字符级分词器:
    ['xxbos', '这', '部', '电', '影', '非', '常', '精', '彩', '，', '演', '员', '的', '表', '演', '令', '人', '印', '象', '深', '刻', '。']


## 为语言模型准备数据


```python
# 将我们的示例数据保存到磁盘供fastai读取
# 在实际应用中，您将使用自己的数据集

# 创建目录
path = Path('chinese_reviews')
path.mkdir(exist_ok=True)
(path/'train').mkdir(exist_ok=True)
(path/'test').mkdir(exist_ok=True)
(path/'train'/'positive').mkdir(exist_ok=True)
(path/'train'/'negative').mkdir(exist_ok=True)
(path/'test'/'positive').mkdir(exist_ok=True)
(path/'test'/'negative').mkdir(exist_ok=True)

# 将数据分为训练集和测试集
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# 保存文件
for i, row in train_df.iterrows():
    with open(path/'train'/row['label']/f"{i}.txt", 'w', encoding='utf-8') as f:
        f.write(row['text'])
        
for i, row in test_df.iterrows():
    with open(path/'test'/row['label']/f"{i}.txt", 'w', encoding='utf-8') as f:
        f.write(row['text'])
```

## 为中文创建自定义TextBlock


```python
# 为中文创建自定义TextBlock
class ChineseTextBlock(TextBlock):
    @delegates(TextBlock.__init__)
    def __init__(self, tokenizer_func=chinese_word_tokenizer, vocab=None, is_lm=False, seq_len=72, **kwargs):
        # 创建分词器转换
        tok_tfm = ChineseTokenizer(tokenizer_func)
        # 将分词器传递给父类
        super().__init__(tok_tfm=tok_tfm, vocab=vocab, is_lm=is_lm, seq_len=seq_len, **kwargs)
        self.tokenizer = tok_tfm
    
    def get_tokenizer(self, **kwargs):
        return self.tokenizer
```

## 为分类创建DataLoaders


```python
# 为分类创建DataLoaders
chinese_block = ChineseTextBlock(tokenizer_func=chinese_word_tokenizer, is_lm=False)

dls = DataBlock(
    blocks=(chinese_block, CategoryBlock),
    get_items=get_text_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, bs=4)  # 为我们的小数据集使用小批量大小

dls.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>xxbos 剧情 xxunk ， 特效 xxunk ， 是 今年 xxunk xxunk 的 电影 xxunk 。</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>xxbos 导演 xxunk 不 xxunk xxunk xxunk xxunk xxunk ， xxunk 电影 xxunk xxunk 。</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xxbos 剧情 xxunk ， 特效 xxunk ， 是 今年 xxunk xxunk 的 电影 xxunk 。</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xxbos 导演 xxunk 不 xxunk xxunk xxunk xxunk xxunk ， xxunk 电影 xxunk xxunk 。</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>


## 构建简单的中文文本分类器

为了演示目的，我们将构建一个简单的分类器。在有更多数据的实际应用中，您将遵循ULMFiT方法进行语言模型预训练。


```python
# 创建一个简单的文本分类器
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# 训练几个周期（对于我们的小数据集，这只是为了演示）
learn.fit_one_cycle(10, 1e-2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.740130</td>
      <td>0.703697</td>
      <td>0.250000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.786076</td>
      <td>0.737464</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.756238</td>
      <td>0.728642</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.760603</td>
      <td>0.852913</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.702520</td>
      <td>0.872675</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.740408</td>
      <td>0.778970</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.748301</td>
      <td>0.836783</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.760259</td>
      <td>0.835310</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.743047</td>
      <td>0.804637</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.762739</td>
      <td>0.820280</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python

```

## 进行预测


```python
# 创建一个辅助函数来预测新文本
def predict_chinese_text(learner, text):
    """辅助函数，用于预测新中文文本的情感"""
    # 创建一个带有文本的临时文件
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name
    
    try:
        # 使用文件路径进行预测（fastai可以处理）
        pred_class, pred_idx, probs = learner.predict(Path(temp_path))
        return pred_class, pred_idx, probs
    finally:
        # 清理临时文件
        os.unlink(temp_path)

# 在新评论上测试
new_review = "这部电影情节紧凑，演员演技精湛，非常推荐！"
pred_class, pred_idx, probs = predict_chinese_text(learn, new_review)
print(f"预测: {pred_class}")
print(f"概率: {probs[pred_idx]:.4f}")
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





    预测: positive
    概率: 0.5288


## 比较词级与字符级分词

让我们比较词级与字符级分词对中文的性能。


```python
# 使用字符级分词创建DataLoaders
char_block = ChineseTextBlock(tokenizer_func=chinese_char_tokenizer, is_lm=False)

char_dls = DataBlock(
    blocks=(char_block, CategoryBlock),
    get_items=get_text_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, bs=4)

# 使用字符级分词创建分类器
char_learn = text_classifier_learner(char_dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# 训练相同数量的周期
char_learn.fit_one_cycle(10, 1e-2)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.789537</td>
      <td>0.688798</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.885231</td>
      <td>0.698306</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.906784</td>
      <td>0.656616</td>
      <td>0.750000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.794850</td>
      <td>0.701583</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.793879</td>
      <td>0.680373</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.811998</td>
      <td>0.583346</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.869714</td>
      <td>0.567899</td>
      <td>0.750000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.895559</td>
      <td>0.562752</td>
      <td>0.750000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.891531</td>
      <td>0.557563</td>
      <td>0.750000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.883717</td>
      <td>0.533585</td>
      <td>0.750000</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
# 比较预测
new_review = "这部电影情节紧凑，演员演技精湛，非常推荐！"

# 词级预测
word_pred_class, word_pred_idx, word_probs = predict_chinese_text(learn, new_review)
print(f"词级预测: {word_pred_class}")
print(f"词级概率: {word_probs[word_pred_idx]:.4f}")

# 字符级预测
char_pred_class, char_pred_idx, char_probs = predict_chinese_text(char_learn, new_review)
print(f"字符级预测: {char_pred_class}")
print(f"字符级概率: {char_probs[char_pred_idx]:.4f}")
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





    词级预测: positive
    词级概率: 0.5288



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





    字符级预测: negative
    字符级概率: 0.5670


## 使用预训练中文模型（高级）

对于生产应用，您通常会使用预训练模型。以下是如何使用transformers库集成预训练的中文BERT模型。


```python
# 如果您安装了transformers库，取消注释并运行此代码
# !pip install transformers

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # 加载预训练的中文BERT
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# # 对示例文本进行分词
# inputs = tokenizer(new_review, return_tensors="pt")

# # 获取预测
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     print(predictions)
```

## 结论

在本教程中，我们展示了如何将fastai的NLP功能适应中文文本处理。我们探索了：

1. 中文的不同分词方法（词级与字符级）
2. 为fastai创建自定义分词器
3. 构建简单的中文文本分类器
4. 比较不同方法

对于有更大数据集的实际应用，您将遵循完整的ULMFiT方法：
1. 在大型中文语料库上预训练语言模型
2. 在特定领域数据上微调语言模型
3. 使用语言模型微调分类器

您也可能使用更先进的模型，如中文BERT、RoBERTa或MacBERT，以获得最先进的性能。