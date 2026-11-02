---
layout: post
title: "Pretraining from Scratch: Loss, Training Loop, and Decoding"
date: 2026-10-31
comments: true
tags:
  - pretraining
  - transformers
  - LLM
  - PyTorch
  - deep learning
description: "First-stage pretraining for a causal LM: next-token targets, cross-entropy loss, train vs validation curves, the training loop, then temperature and top-k at generation time."
glossary:
  - Pretraining
  - Next-Token Prediction
  - Logit
  - Softmax
  - Cross-Entropy
  - Loss Function
  - Overfitting
  - Temperature
  - Top-k
  - Decoding
  - Epoch
  - Batch Size
  - Optimizer
  - Inference
  - Causal Language Model
  - Base Model
  - Open-Source Model
  - Embedding
  - LM Head
  - Self-Supervised Learning
  - Prompt
---

[**Pretraining**](/ai-glossary/pretraining/) is the first training stage of a language model. You start from random weights (or a small scratch run), train on a large text corpus with [**next-token prediction**](/ai-glossary/next-token-prediction/), and end up with a [**base model**](/ai-glossary/base-model/) that can continue text. [**Open-source models**](/ai-glossary/open-source-model/) such as Qwen already finished this stage. Specializing an open model for one product domain is a later step ([continued pretraining](/ai-glossary/continued-pretraining/) or fine-tuning), which is a different post.

This post covers the first-stage mechanism:
1. What pretraining is (and is not)
2. An untrained model on a concrete prompt
3. The learning signal: input tokens, target tokens, [logits](/ai-glossary/logit/), probabilities, [cross-entropy](/ai-glossary/cross-entropy/)
4. Training loss vs validation loss
5. The training loop
6. [Decoding](/ai-glossary/decoding/) after training: greedy, [temperature](/ai-glossary/temperature/), [top-k](/ai-glossary/top-k/)
7. Saving and loading weights
8. Takeaways

Some architectural background knowledge is assumed:
- [embeddings](/ai-glossary/embedding/), 
- [causal self-attention](/2026/10/29/attention-mechanism-from-scratch.html), 
- [final vocab projection](/ai-glossary/lm-head/)


## 1. What pretraining is

A [causal language model](/ai-glossary/causal-language-model/) is trained with one job: given the words so far, predict the **next** word.

Example. The text is:

> `the cat sat on the mat`

At training time the model sees prefixes and must assign high probability to the true continuation:

| Given so far | Should prefer |
|:-------------|:--------------|
| `the` | `cat` |
| `the cat` | `sat` |
| `the cat sat` | `on` |
| `the cat sat on` | `the` |
| `the cat sat on the` | `mat` |

The text itself is the label. Nobody wrote a separate answer key. That setup is called [**self-supervised**](/ai-glossary/self-supervised-learning/): the next word in the corpus is the target. For a causal language model, that **is** [pretraining](/ai-glossary/pretraining/): run next-token prediction over a large collection of ordinary text. The question at every position is the same: given the words so far, how probable is the true next token?

When this stage finishes, you have a [**base model**](/ai-glossary/base-model/): weights that can continue ordinary text by predicting the next token.

## 2. Untrained model (generation mode)

Before any training, initialize the model and ask it to continue a [**prompt**](/ai-glossary/prompt/). This is **generation** ([inference](/ai-glossary/inference/)): you feed the model its own previous outputs.

Prompt:

> `Rain falls`

An untrained model might continue with:

> `Rain falls quilt zebra the the mat sat`

The final layer outputs nearly random [logits](/ai-glossary/logit/) over the vocabulary. Softmax turns them into a near-uniform distribution. Sampling then draws tokens with little structure.

We need a number that says how wrong the next-token prediction is, so gradient descent can update the weights. That number is the [**loss**](/ai-glossary/loss-function/).

## 3. The learning signal (training mode)

Stay in **training** mode for this whole section. We have the full sentence. We do **not** feed the model its own mistakes as the next input. We feed the real previous tokens and score the prediction against the real next token.

### 3.1 A toy vocabulary

Use a five-word vocabulary so every vector fits on one screen:

| id | token |
|---:|:------|
| 0 | `the` |
| 1 | `cat` |
| 2 | `sat` |
| 3 | `on` |
| 4 | `mat` |

Sentence: `the cat sat on mat`  
Token ids: `[0, 1, 2, 3, 4]`

### 3.2 Inputs and targets are the same sequence, shifted

For language modeling we build two aligned sequences:

```text
tokens:  the   cat   sat   on   mat
ids:      0     1     2    3    4

input:   [0, 1, 2, 3]     # the cat sat on
target:  [1, 2, 3, 4]     # cat sat on mat
```

Position by position:

| Step | Input context (what the model sees) | Target id | Target word |
|-----:|:------------------------------------|----------:|:------------|
| 0 | `the` | 1 | `cat` |
| 1 | `the cat` | 2 | `sat` |
| 2 | `the cat sat` | 3 | `on` |
| 3 | `the cat sat on` | 4 | `mat` |

Same idea in code:

```python
tokens = torch.tensor([0, 1, 2, 3, 4])
inputs = tokens[:-1]   # [0, 1, 2, 3]
targets = tokens[1:]   # [1, 2, 3, 4]
```

**Training vs generating, same architecture:**

| Mode | What goes into position $t$ | What we compare against |
|:-----|:----------------------------|:------------------------|
| **Training** | The **true** token from the corpus at $t$ | The true next token (target) |
| **Generating** | The token the model **just produced** at $t-1$ | Nothing. We sample or argmax instead |

If training fed the model its own wrong guesses the way generation does, an early error would enter the input history and distort later loss terms. Training uses the real history so every position asks the same question: given this real prefix, how wrong is the next-token prediction?

### 3.3 From context to logits

The model reads `inputs` and returns a [logit](/ai-glossary/logit/) vector **per position**. Each logit vector has one score per vocabulary word. With vocab size 5:

```text
logits: (sequence_length, vocab_size) = (4, 5)
```

Invented numbers for position 0 (context = `the`, target should be `cat` = id 1):

| token | id | logit (made up) |
|:------|---:|----------------:|
| `the` | 0 | 0.40 |
| `cat` | 1 | 0.10 |
| `sat` | 2 | 0.30 |
| `on`  | 3 | 0.05 |
| `mat` | 4 | 0.15 |

Before training, the highest logit might not be the target. Here `the` (0.40) beats `cat` (0.10). The model currently prefers repeating `the`.

### 3.4 Softmax turns logits into probabilities

Apply [softmax](/ai-glossary/softmax/) over the five logits. Softmax makes them positive and sum to 1:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Rough probabilities for the table above (rounded):

| token | $p$ |
|:------|----:|
| `the` | 0.26 |
| `cat` | 0.19 |
| `sat` | 0.23 |
| `on`  | 0.18 |
| `mat` | 0.19 |

The target is `cat`. The model assigned it about $0.19$ probability.

### 3.5 Cross-entropy: low probability on the true next token means high loss

For one position, [cross-entropy](/ai-glossary/cross-entropy/) loss is:

$$
\mathcal{L} = -\log p_{\text{target}}
$$

If $p_{\text{cat}} = 0.19$, then $\mathcal{L} \approx -\log(0.19) \approx 1.66$.

If training later raises $p_{\text{cat}}$ to $0.80$, then $\mathcal{L} \approx -\log(0.80) \approx 0.22$. Lower is better.

Loss is $-\log$ of the probability assigned to the true next token. When that probability is low, loss is high.

Over a sequence we average (or sum) that number across positions. Over a [batch](/ai-glossary/batch-size/) we average across examples too.

In PyTorch the usual call is:

```python
# logits: (batch, time, vocab), targets: (batch, time)
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

`cross_entropy` expects raw logits. It applies log-softmax internally for numerical stability. Do not softmax first and then call it.

### 3.6 What training actually changes

Backpropagation updates every weight so that, on these prefixes, the logit for the true next token rises relative to the others. After many batches of many documents, next-token probability on the training distribution improves.

The model does not store a separate fact table. Patterns that help next-token prediction (for example cities after "capital of", or code after `def`) show up as higher probability for those continuations.

## 4. Training loss vs validation loss

Compute the **same** cross-entropy on two different slices of text.

**Training set.** Books, web pages, or whatever you chose to learn from. The optimizer sees these batches.

**Validation set.** Text held out from training. Same loss formula. The optimizer never steps on these examples.

Concrete split for a tiny corpus of 10 lines:

```text
lines 1-8  → training
lines 9-10 → validation
```

Plot both curves every few steps:

```text
step    train_loss    val_loss
   0        4.10         4.12     # near log(vocab) if random
 100        2.40         2.55
 500        1.10         1.45
1000        0.40         1.80     # train still falling, val rising
```

When training loss keeps falling while validation loss rises, the model is fitting training-set quirks instead of structure that transfers. That is [**overfitting**](/ai-glossary/overfitting/). Early stopping, more data, [dropout](/ai-glossary/dropout/), and weight decay are common responses. For this post, the point is to **read both curves**, not only the training number.

## 5. The training loop

One [epoch](/ai-glossary/epoch/) means one pass over the training data. Inside an epoch you walk batches. For each batch:

1. Take a batch of `inputs` and `targets` (shifted as above).
2. Forward: `logits = model(inputs)`.
3. Compute `loss` with cross-entropy.
4. `loss.backward()` to get gradients.
5. [Optimizer](/ai-glossary/optimizer/) step (commonly AdamW) to update weights.
6. Zero gradients for the next batch.
7. Occasionally evaluate validation loss and print a short generation sample.

Compact sketch (architecture omitted on purpose):

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

What each line does:

- `model.train()` turns on dropout (and similar) behavior used while learning.
- The loader yields real prefixes and real next tokens from the corpus.
- `backward` + `step` update weights so the true next token becomes more probable on the next pass.

After enough steps, switch to `model.eval()`, turn off dropout, and generate from a prompt. That is a different mode with different inputs (see the table in §3.2).

## 6. Decoding: turning logits into text after training

Training optimized next-token probabilities. **Decoding** decides how to pick the next token at generation time. These settings do not change the weights. They only change how you sample from the distribution the model already learned.

Stay with the five-word toy. Suppose after training, given `the cat sat on`, the model’s probabilities for the next token are:

| token | $p$ |
|:------|----:|
| `the` | 0.05 |
| `cat` | 0.05 |
| `sat` | 0.10 |
| `on`  | 0.10 |
| `mat` | 0.70 |

### 6.1 Greedy decoding

Always take $\arg\max_i p_i$. Here that is `mat`. Greedy is deterministic. On real models it often repeats or sticks to high-probability tokens, because it always follows the single highest peak.

### 6.2 Temperature

[Temperature](/ai-glossary/temperature/) $T$ rescales logits **before** softmax:

$$
p_i = \mathrm{softmax}(z_i / T)
$$

Same logits, different $T$:

| Behavior | $T$ | Effect on the toy |
|:---------|----:|:------------------|
| Sharper | $0.5$ | `mat` becomes more dominant than $0.70` |
| Default | $1.0$ | Distribution unchanged |
| Flatter | $1.5$ | `mat` loses probability mass; `sat` / `on` get more |

Low $T$ concentrates mass on the top tokens. High $T$ raises the chance of lower-probability tokens. Whether that helps or hurts depends on the prompt and the trained model.

### 6.3 Top-k

[Top-k](/ai-glossary/top-k/) keeps only the $k$ largest logits, sets the rest to $-\infty$ (probability 0), then applies softmax and samples.

With $k=2$ on the table above, only `mat` and (say) `sat` remain. Tokens like `the` cannot be chosen at this step even if temperature is high.

Typical combo at generation time: apply top-k (or top-p), optionally scale by temperature, then sample. Training still used plain cross-entropy on the full vocabulary. These filters are an **inference** choice.

### 6.4 Tiny generate loop

```python
model.eval()
idx = text_to_ids("the cat sat on")  # (1, T)

for _ in range(max_new_tokens):
    logits = model(idx)[:, -1, :]          # last position only
    logits = logits / temperature
    # optional: keep only top-k logits, mask the rest
    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_id], dim=1)
```

Each iteration appends one token and feeds the growing sequence back in. An early bad sample stays in the input history for later steps. Training never had that problem, because it always fed true prefixes.

## 7. Saving and loading weights

Training takes wall-clock time. You want to pause and resume without starting from random weights again.

```python
# save
torch.save(model.state_dict(), "model.pt")

# load into the same architecture
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

`state_dict` is only the tensors. The class definition (your GPT module) must match when you load. You can also save optimizer state if you plan to continue training, not only run inference.

Public models (Qwen and others) are the output of this same objective at large scale. Loading their weights skips the expensive first stage. What you do next on your own corpus is not this post.

## Summary

```text
1. Untrained GPT          → near-uniform next-token probs → unstructured text
2. Shifted targets        → every position has a true next id
3. Softmax + cross-entropy → loss = -log p(true next)
4. Train vs val curves    → detect overfitting
5. Optimizer loop         → weights move to raise p(true next)
6. Decoding settings      → temperature / top-k change sampling
7. Save / load weights    → pause, resume, or reuse a public base
```

| Stage | Concrete question it answers |
|:------|:-----------------------------|
| Untrained sample | Why bother training at all? |
| Input / target shift | What is the label for each position? |
| Cross-entropy | How wrong was the guess for `cat` after `the`? |
| Val loss | Are we memorizing line 3 of the train file? |
| Train loop | How do batches update weights? |
| Temperature / top-k | Why does the same model produce different text in a chat UI? |
| Save / load | How do we avoid throwing work away? |

## Takeaways

- Pretraining is the first stage: next-token prediction on unlabeled text until you have a base model.
- Training feeds true prefixes and scores true next tokens. Generation feeds the model's own outputs back in.
- Loss is $-\log p$ of the correct next token, averaged over positions and batches.
- Watch validation loss, not only training loss.
- Temperature and top-k change sampling after weights are fixed. They are not a substitute for training.
- Saved weights let you resume training or load a public base model later.

Next on this site: other from-scratch pieces in the series. Domain specialization of an open model (continued pretraining on product text) belongs with the product research notes, not in this post.

## References

1. <span id="ref-radford-gpt2"></span>Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. “Language Models are Unsupervised Multitask Learners.” OpenAI technical report, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf){:target="_blank" rel="noopener noreferrer"}.
2. <span id="ref-vaswani"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need.” In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762){:target="_blank" rel="noopener noreferrer"}.  
   Read in [BetterReads](https://www.betterreads.dev/reader?bookId=vaswani_attention_is_all_you_need&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
3. Zhao Lü. [*Attention from Scratch: Soft Attention to Multi-Head*](/2026/10/29/attention-mechanism-from-scratch.html){:target="_blank" rel="noopener noreferrer"}. Causal attention background for this post.
