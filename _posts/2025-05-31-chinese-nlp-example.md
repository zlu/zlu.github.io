---
layout: post
title: "Chinese Natural Language Processing with fastai - Fastai Part 4"
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
description: "Chinese Natural Language Processing with fastai - Fastai Part 4"
---
## NLP with fastai

In the previous tutorial we have seen how to leverage pretrained model and fine tune it to perform categorization tasks on images (MNIST).  The principle of transfer learning is applied there can also be applied to NLP tasks.  In this tutorial, we will use a pre-trained model calld AWD_LSTM to classify Chinese movie reviews.  AWD_LSTM is a variant of LSTM.  LSTM is a type of recurrent neural network (RNN) that is designed to handle long sequences of text.  We will leave detailed discussion on RNNs to a later tutorial.

## Chinese NLP with fastai: Practical Example

Chinese language processing is a challenging task because majority of NLP models are trained with western languages such as English.  Unlike English, Chinese does not use spaces to separate words.  This makes tokenization more challenging. Lukily there are libraries like jieba for Chinese tokenization.  Jieba and pkuseg are two libraries designed to handle Chinese segmentation effectively.  Pre-trained word embeddings such as Word2Vec, Glove, or FastText can be used as long as they are trained on Chinese corpora. Towards the end of this guide, I will show you how to use Google's BERT variant, Chinese BERT, to capture the context in Chinese text.  XLM-RoBERTa is another multilingual model that performs well on Chinese text.  Besides Chinese BERT, there are many local-grown models such as ERNIE (Enhanced Representation through kNowledge IntEgration) and PaddleNLP from Baidu, FastBERT and AliceMind from Alibaba, and last but not least, TecentPretrain and Chinese Word Vectors from Tencent.

## The Process
Largely speaking, there are two basic blocks to NLP tasks: text preprocessing and text classification.

In text preprocessing, we want to prepare the text in such a way that computer is able to interpret it.  Interpretation of the contextual meaning of the texts turns out to be a non-trivial task even for RNNs.  The introduction of transformer and self-attention had made a breakthrough in this area (hence the transformer example at the end).  For the sake of simplicity, we will mainly focus now on **tokenization** and **word embeddings** steps.

### Tokenization

Tokenization is the action of converting text into "tokens", which could be characters ("a", "b", "c", ...) or words ("hello", "world", ...), or even substrings depending on the granularity of the model.  This is where Chinese language gets interesting, as unlike English or alphabet-based languages, even Chinese character (我，喜，欢，爱，中) carry meanings of their own!  Word segmentation in Chinese thus becomes a tougher task as unlike English, words are separated by spaces, Chinese people had to learn how to spot word boundaries by reading and memorization.  Special algorithms are thus needed to segment Chinese text.  In addition, foreign words, numbers, and symbols in Chinese texts require special handling.

### Word Embeddings
Word embeddings are a way to represent words as vectors.  In the last tutorial, we saw how to convert MNIST dataset (grayscale images) into 3D vectors (height, width, color).  We will do something here quite similar conceptually.  What's special about these vectors is that they are learned from a large corpus of text and those similar in meanings are close to each other in a high-dimensional vector space (物以类聚).  In this tutorial, we will use create a custom fastai `DataBlock`, `ChineseTextBlock`, to tokenize and embed Chinese text.

### Text Classification
Text classification is the task of assigning a label to a piece of text.  For example, we can classify a movie review as positive or negative.  We  will use fastai's dataloaders and `AWD_LSTM` to build a text classifier.

## Setup and Imports


```python
# Install required packages if needed
# !pip install fastai jieba
```


```python
from fastai.text.all import *
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Sample Chinese Text

For demonstration purposes, we'll create a small dataset of Chinese movie reviews. In a real application, you would load your own dataset.


```python
# Sample positive and negative movie reviews in Chinese
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

# Create a DataFrame
reviews = positive_reviews + negative_reviews
labels = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews)

df = pd.DataFrame({'text': reviews, 'label': labels})
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

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



## Chinese Text Tokenization

Let's explore different tokenization methods for Chinese text.

### 1. Word-level Tokenization with Jieba

We need to distinguish the meaning of `word` in the context of NLP for Chinese.  A Chinese word is composed of Chinese characters that provide meaning.  For example, the word `中国` is composed of two Chinese characters `中` and `国`.  The word `中国` has a different meaning than the word `中国` in `中国是一个伟大的国家` (the country China is a great country).  The word `中国` in the latter sentence is a noun phrase, while the word `中国` in the former sentence is a noun.  The word `中国` in the former sentence is a single word, while the word `中国` in the latter sentence is two words.  The word `中国` in the former sentence is a single token, while the word `中国` in the latter sentence is two tokens. 
In English however, a word is a a word like `China`.  So a Chinese NLP word is really conceptually more similar to a `subword` in English.


```python
def chinese_word_tokenizer(text):
    """Tokenize Chinese text using Jieba word segmentation"""
    # Handle Path objects by reading the file
    if hasattr(text, 'read_text'):
        text = text.read_text(encoding='utf-8')
    elif hasattr(text, 'open'):
        text = text.open(encoding='utf-8').read()
    
    # Convert to lowercase if there's any English text mixed in
    text = str(text).lower()
    # Use Jieba to segment words
    words = jieba.cut(text)
    return list(words)

# Example
sample_text = "这部电影非常精彩，演员的表演令人印象深刻。"
word_tokens = chinese_word_tokenizer(sample_text)
print(f"Word tokens: {word_tokens}")
print(f"Number of tokens: {len(word_tokens)}")
```

    Word tokens: ['这部', '电影', '非常', '精彩', '，', '演员', '的', '表演', '令人', '印象', '深刻', '。']
    Number of tokens: 12


### 2. Character-level Tokenization


```python
def chinese_char_tokenizer(text):
    """Tokenize Chinese text at character level"""
    # Handle Path objects by reading the file
    if hasattr(text, 'read_text'):
        text = text.read_text(encoding='utf-8')
    elif hasattr(text, 'open'):
        text = text.open(encoding='utf-8').read()
    
    # Convert to string if it's not already
    text = str(text)
    # Remove spaces if any
    text = text.replace(" ", "")
    # Split into characters
    return list(text)

# Example
char_tokens = chinese_char_tokenizer(sample_text)
print(f"Character tokens: {char_tokens}")
print(f"Number of tokens: {len(char_tokens)}")
```

    Character tokens: ['这', '部', '电', '影', '非', '常', '精', '彩', '，', '演', '员', '的', '表', '演', '令', '人', '印', '象', '深', '刻', '。']
    Number of tokens: 21


### 3. Custom fastai Tokenizer for Chinese

In fastai's NLP framework, special tokens play a crucial role in helping models understand text structure.  In the code snippet below, you will see `xxbos', which tells the model that a new sentence is starting. Some other often used special tokens include `xxmaj` (for capitalization), `xxup` (for uppercase), `xxmaj` (for uppercase), `xxrep` (for repeating a word), and `xxwrep` (for repeating a word with a space in between).



```python
# Define a string class that can be truncated for display
class TitledStr(str):
    """A string that can be truncated for display purposes"""
    def truncate(self, n):
        return TitledStr(self[:n] + '...' if len(self) > n else self)
        
    def show(self, ctx=None, **kwargs):
        "Display text in the context"
        return show_text(self, ctx=ctx, **kwargs)
        
def show_text(text, ctx=None, **kwargs):
    "Helper function to display text"
    if ctx is None: ctx = {'text': text}
    else: ctx['text'] = text
    return ctx

class ChineseTokenizer(Transform):
    def __init__(self, tokenizer_func=chinese_word_tokenizer):
        self.tokenizer_func = tokenizer_func
        
    def encodes(self, x):
        tokens = self.tokenizer_func(x)
        # Add special tokens like BOS (beginning of sentence)
        tokens = ['xxbos'] + tokens
        return tokens
    
    def decodes(self, x):
        text = ''.join(x) if isinstance(x[0], str) and len(x[0]) == 1 else ' '.join(x)
        # Create a text object with a truncate method
        return TitledStr(text)

# Create instances for both tokenization methods
word_tokenizer = ChineseTokenizer(chinese_word_tokenizer)
char_tokenizer = ChineseTokenizer(chinese_char_tokenizer)

# Example
print("Word tokenizer:")
print(word_tokenizer.encodes(sample_text))
print("\nCharacter tokenizer:")
print(char_tokenizer.encodes(sample_text))
```

    Word tokenizer:
    ['xxbos', '这部', '电影', '非常', '精彩', '，', '演员', '的', '表演', '令人', '印象', '深刻', '。']
    
    Character tokenizer:
    ['xxbos', '这', '部', '电', '影', '非', '常', '精', '彩', '，', '演', '员', '的', '表', '演', '令', '人', '印', '象', '深', '刻', '。']


## Preparing Data for Language Model


```python
# Save our sample data to disk for fastai to read
# In a real application, you would use your own dataset

# Create directories
path = Path('chinese_reviews')
path.mkdir(exist_ok=True)
(path/'train').mkdir(exist_ok=True)
(path/'test').mkdir(exist_ok=True)
(path/'train'/'positive').mkdir(exist_ok=True)
(path/'train'/'negative').mkdir(exist_ok=True)
(path/'test'/'positive').mkdir(exist_ok=True)
(path/'test'/'negative').mkdir(exist_ok=True)

# Split data into train and test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Save files
for i, row in train_df.iterrows():
    with open(path/'train'/row['label']/f"{i}.txt", 'w', encoding='utf-8') as f:
        f.write(row['text'])
        
for i, row in test_df.iterrows():
    with open(path/'test'/row['label']/f"{i}.txt", 'w', encoding='utf-8') as f:
        f.write(row['text'])
```

## Creating a Custom TextBlock for Chinese


```python
# Create a custom TextBlock for Chinese
class ChineseTextBlock(TextBlock):
    @delegates(TextBlock.__init__)
    def __init__(self, tokenizer_func=chinese_word_tokenizer, vocab=None, is_lm=False, seq_len=72, **kwargs):
        # Create the tokenizer transform
        tok_tfm = ChineseTokenizer(tokenizer_func)
        # Pass the tokenizer to the parent class
        super().__init__(tok_tfm=tok_tfm, vocab=vocab, is_lm=is_lm, seq_len=seq_len, **kwargs)
        self.tokenizer = tok_tfm
    
    def get_tokenizer(self, **kwargs):
        return self.tokenizer
```

## Creating DataLoaders for Classification


```python
# Create DataLoaders for classification
chinese_block = ChineseTextBlock(tokenizer_func=chinese_word_tokenizer, is_lm=False)

dls = DataBlock(
    blocks=(chinese_block, CategoryBlock),
    get_items=get_text_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, bs=4)  # Small batch size for our small dataset

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


## Building a Simple Chinese Text Classifier

For demonstration purposes, we'll build a simple classifier. In a real application with more data, you would follow the ULMFiT approach with language model pretraining.


```python
# Create a simple text classifier
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# Train for a few epochs (with our tiny dataset, this is just for demonstration)
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

## Making Predictions


```python
# Create a helper function to predict on new text
def predict_chinese_text(learner, text):
    """Helper function to predict sentiment on new Chinese text"""
    # Create a temporary file with the text
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_path = f.name
    
    try:
        # Use the file path for prediction (which fastai can handle)
        pred_class, pred_idx, probs = learner.predict(Path(temp_path))
        return pred_class, pred_idx, probs
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

# Test on a new review
new_review = "这部电影情节紧凑，演员演技精湛，非常推荐！"
pred_class, pred_idx, probs = predict_chinese_text(learn, new_review)
print(f"Prediction: {pred_class}")
print(f"Probability: {probs[pred_idx]:.4f}")
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







    Prediction: positive
    Probability: 0.5288


## Comparing Word vs. Character Tokenization

Let's compare the performance of word-level vs. character-level tokenization for Chinese.


```python
# Create DataLoaders with character-level tokenization
char_block = ChineseTextBlock(tokenizer_func=chinese_char_tokenizer, is_lm=False)

char_dls = DataBlock(
    blocks=(char_block, CategoryBlock),
    get_items=get_text_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, bs=4)

# Create a classifier with character-level tokenization
char_learn = text_classifier_learner(char_dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# Train for the same number of epochs
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
# Compare predictions
new_review = "这部电影情节紧凑，演员演技精湛，非常推荐！"

# Word-level prediction
word_pred_class, word_pred_idx, word_probs = predict_chinese_text(learn, new_review)
print(f"Word-level prediction: {word_pred_class}")
print(f"Word-level probability: {word_probs[word_pred_idx]:.4f}")

# Character-level prediction
char_pred_class, char_pred_idx, char_probs = predict_chinese_text(char_learn, new_review)
print(f"Character-level prediction: {char_pred_class}")
print(f"Character-level probability: {char_probs[char_pred_idx]:.4f}")
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







    Word-level prediction: positive
    Word-level probability: 0.5288




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







    Character-level prediction: negative
    Character-level probability: 0.5670


## Using Pre-trained Chinese Models (Advanced)

For production applications, you would typically use pre-trained models. Here's how you might integrate a pre-trained Chinese BERT model using the transformers library.


```python
# Uncomment and run this if you have the transformers library installed
# !pip install transformers

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# # Load pre-trained Chinese BERT
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

# # Tokenize a sample text
# inputs = tokenizer(new_review, return_tensors="pt")

# # Get predictions
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     print(predictions)
```

## Conclusion

In this notebook, we've demonstrated how to adapt fastai's NLP capabilities for Chinese text processing. We've explored:

1. Different tokenization methods for Chinese (word-level vs. character-level)
2. Creating custom tokenizers for fastai
3. Building a simple Chinese text classifier
4. Comparing different approaches

For real-world applications with larger datasets, you would follow the complete ULMFiT approach:
1. Pre-train a language model on a large Chinese corpus
2. Fine-tune the language model on your domain-specific data
3. Fine-tune a classifier using the language model

You would also likely use more advanced models like Chinese BERT, RoBERTa, or MacBERT for state-of-the-art performance.
