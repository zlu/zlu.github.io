---
layout: post
title: "Succinct Guide Guide on Text Classification with fastai - Fastai Part 5"
date: 2025-05-30
comments: true
tags:
  - python
  - artificial intelligence
  - machine learning
  - fastai
  - PyTorch
description: "Succinct Guide Guide on Text Classification with fastai - Fastai Part 5"
---
## Subword Tokenization
For Chinese, where the concept of "word" does not exist, fastai's `subword` support via `SentencePiece` becomes crucial.  As there is no whitespace between words:
```
我喜欢学习 （I like studying)
```
SentencePiece will segment it into:
```
我/喜欢/学习/
```
This is done via an unsupervised learning process directly on raw Chinese text, which allows the model to generalize to new words and expressions.  This is important for Chinese because it allows the model to learn the correct word boundaries and avoid splitting words in unexpected ways.  As a result, we could apply them to AWD-LSTM or Transformer-based models used in fastai's NLP pipeline.  
Here's another example:
```
我喜欢吃辣椒。
▁我 ▁喜欢 ▁吃 ▁辣 ▁椒 ▁。
```
Each token is either a character or a frequent combination, which the model has learned over time from the training data (raw text).  The `_` marks the beginnings of a new subword.  As you can see, it has correctly grouped `喜` (happiness) and `欢` (joy) as a single token `喜欢` (like).

Specifically, subword tokenization can be done like this:

```python
# Suppose `text` is the raw text
def subword(vocab_size):
  sw = SubwordTokenizer(vocab_size=vocab_size)
  sw.setup(text)
  return ' '.join(first(sw([text])))
```

Note, depending on the size of the vocabulary, the subword tokenization will likely yield different results.  The larger the vocabulary, the fewer tokens per sentence, the faster the training time, but also a larger embedding matrix.  This is why we need to find a balance.

## Numericalization - Turning Tokens into Numbers.

In part 2 of this series, we have learned how to turn images into numbers for categorization tasks.  The same principle applies to text.  Computers work only with numbers, so we need to turn tokens from the previous step into numbers. Then we can feed them into a neural network.

We will leverage fastai's `Numericalize()` to transform the tokens into integers.  This is done by creating a `Vocab` object, which is a mapping of tokens to integers.  The `Numericalize()` will then use this mapping to transform the tokens into integers.  Then we can feed them into a fastai `Datasets` object, which applies the same transformation to the whole dataset.  The resulting `dataset.items` will contain the integers.

Here's a toy example:

```python
from fastai.text.all import *
from fastcore.basics import noop

tokens = [["I", "love", "deep", "learning"], ["Fastai", "makes", "it", "simple"]]

# Apply noop + numericalize
dsets = Datasets(tokens, [[noop, Numericalize()]])

# Show vocab
vocab = dsets[0][0].vocab
print("Vocab:\n", vocab)

# Show numericalized data
for i, item in enumerate(dsets):
    print(f"Sentence {i+1}: {item[0]}")
```

And let's visualize what's happening:

```python
Tokenized Text → [ "Fastai", "makes", "it", "simple" ]
                   ↓
               Numericalize
                   ↓
Integer IDs    → [ 6, 7, 8, 9 ]

Where:
    vocab = { "Fastai": 6, "makes": 7, "it": 8, "simple": 9, ... }
```

## DataLoader Creation

We have seen `DataLoader` in the previous parts of this series.  It takes raw or processed data (such as numericalized text from the previous step) and turns it into batches.  This is important for training a neural network because it allows the model to see the data in a batch-wise manner, which is more efficient and stable.  There are two important concepts here to understand:

1. Batching

Neural networks train faster and more reliably with **batches** of data.  So the output from the previous step:
```python
[2, 3, 4, 5]  # "I love deep learning"
[6, 7, 8, 9]  # "Fastai makes it simple"
```
are grouped into a batch.

2. Padding

As sequence data structures are often of variable length and tensors **must** be of the same size (rectangular) to fit into GPU memory, we need to pad the sequences to the same length.  For example, the two variable sequences:
```
Original:         [2, 3, 4, 5]
                  [6, 7]
After padding:    [2, 3, 4, 5]
                  [6, 7, 0, 0]
```                  
will be padded to the same length.

Suppose we have already created a `TextDataLoaders` previously like this:

```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
```

Then we can create a padded batch like this:

```python
x, y = dls.one_batch()
print(type(x))  # torch.Tensor
print(x.shape)  # e.g., torch.Size([64, 72]) — 64 examples, each 72 tokens long
```

## Language Model Fine-tuning

Just as we did in the previous part of this series, we can fine-tune a language model as such:

```python
# Create a language model learner
learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# Fine-tune on your corpus
learn.fine_tune(1)
```

`fine_tune(1)` runs one epoch of training on the learn object.  Here the pre-trained base model is `frozen`, meaning its weights are not updated during training.  This is important because we want to focus only on the `head` (or the newly added classification layers) being trained.  This allows the model to adapt its final layers to the new task without disrupting the pre-trained representations.  Then the entire model is `unfrozen` (including the base model) and _all_ layers are fine-tuned together.  When we pass `1` to `fine_tune`, fastai will run 1 epoch with the base model frozen but skip the unfrozen step.  So we will need to pass in a number greater than 1 to fine-tune the entire model.

## Text Classification

Finally, perform the inference:
```python
learn.predict("I love deep learning")
```

This completes our journey of applying RNNs to text classification.  In the next part of this series, we will dive deeper into how to build a RNN network from scratch!
