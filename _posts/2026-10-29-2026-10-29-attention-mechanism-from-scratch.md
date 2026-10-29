---
layout: post
title: "Attention from Scratch: Soft Attention to Multi-Head"
date: 2026-10-29
comments: true
tags:
  - attention
  - transformers
  - LLM
  - PyTorch
  - deep learning
description: "A step-by-step walk through attention: soft attention, weight-free self-attention, trainable QKV, causal masking, dropout, and multi-head. Embeddings are assumed to be already in hand."
glossary:
  - Sequence Model
  - Hidden State
  - Encoder
  - Decoder
  - Encoder–Decoder
  - Attention
  - Self-Attention
  - Multi-Head Attention
  - RNN
  - Softmax
  - Teacher Forcing
  - MLP
  - Autoregressive
  - Matmul
---

Embeddings turn tokens into vectors. That alone is not enough for a language model: each position still needs a way to compute a weighted combination of other positions' vectors. That mechanism is **attention**.

This post covers:
1. Soft attention (encoder–decoder, then the formula)
2. Weight-free self-attention
3. Trainable Q, K, V (plus a compact `SelfAttention` module)
4. Causal masking
5. Dropout on attention weights (then a compact `CausalSelfAttention`)
6. Multi-head attention
7. Summary diagram and checklist

## Why attention exists

Let's start with a simple task of translating from English to French.

> **Input:** `I love coffee`
>
> **Output:** `J'aime le café`

Before doing this, we have already **trained** on many paired examples (English sentence with their corresponding French translation).  The resulting model has learned such mappings.  At **inference** time we only get English and must predict French.

A [**sequence model**](/ai-glossary/sequence-model/) reads tokens one after another and keeps a running vector of what has been seen so far (a [*hidden state*](/ai-glossary/hidden-state/)). Early neural machine translation (MT) used an [**encoder–decoder**](/ai-glossary/encoder-decoder/) split:

- The [**encoder**](/ai-glossary/encoder/) reads the English sentence left to right. After the last English word, its [hidden state](/ai-glossary/hidden-state/) is a fixed-length vector meant to represent the entire input sentence. That final hidden state is what the decoder receives from the encoder.
- The [**decoder**](/ai-glossary/decoder/) then produces French **one word at a time**. At each step it predicts the *next* French word from (1) that English summary vector and (2) the French words already produced so far.


### Training vs generating (same model, different inputs)

| Mode | What we have | What the decoder does |
|:-----|:-------------|:----------------------|
| **Training** | Both English *and* the reference French | At step $t$, we feed it the *correct* previous French words (`J'aime`, then `le`, …) and ask it to predict the next one. Loss compares that prediction to the reference. ([Teacher forcing](/ai-glossary/teacher-forcing/) is defined below.) |
| **Generating** (inference) | English only | We feed it its *own* previous predictions. Start from a start-of-sentence token; sample or argmax the next word; feed that back in; repeat until end. |


Encoder first (both modes): `I → love → coffee` ⇒ summary $s$.

The interesting part is what happens when the decoder’s **first** prediction is wrong. Suppose it says `"Je"` instead of `"J'aime"`.

**Generating.** There is no reference French available, so an early error stays in the input history:

```text
Decoder step 1: given s + <start>     → predicts "Je"      (wrong)
Decoder step 2: given s + "Je"        → predicts ???       ← input history now contains "Je"
Decoder step 3: given s + "Je" + …    → …
```

**Training.** The reference is `J'aime le café`. We still score the bad `"Je"`, but we do **not** feed it forward:

```text
Decoder step 1: given s + <start>     → predicts "Je"      (wrong; loss vs "J'aime")
Decoder step 2: given s + "J'aime"    → predict "le"       ← reference, not "Je"
Decoder step 3: given s + "J'aime"+"le" → predict "café"
Decoder step 4: given s + … +"café"   → predict <end>
```

Look at **decoder step 2**: generating gets `"Je"`; training gets `"J'aime"`. Same model, same English. Only the French history on the *given* side changed.

That training rule is [**teacher forcing**](/ai-glossary/teacher-forcing/): after each prediction, replace the model's output with the correct reference word before the next step, so one early mistake does not change the inputs for later steps during learning. At generating time, there is no teacher. The model must use its own words as the next inputs.

### The bottleneck, with a picture

Without attention, the encoder finishes the English sentence and passes the decoder **one** fixed-size vector $s$, which is usually its final [hidden state](/ai-glossary/hidden-state/) after `coffee`. From then on, every French word the decoder predicts must be computed from that single vector. There is no access to `I`, `love`, or `coffee` as separate vectors.

```text
Encoder (reads English):

  "I" → "love" → "coffee"
                      │
                      ▼
                 [ one vector s ]   ← ALL of the English must fit in this one vector
                      │               (fixed length, say a few hundred numbers)
                      ▼
Decoder (writes French one word at a time):
  start → "J'aime" → "le" → "café" → end
           ↑________________↑________↑
           each of these steps only sees s, never the English words again
```

This single-vector design is the **bottleneck**:

- $s$ has a **fixed length**, no matter whether the English is 3 words or 30. A long sentence must still fit in the same-sized vector.
- The encoder builds $s$ by reading left to right, so later words overwrite earlier ones inside the hidden state. By the time you reach `coffee`, the contribution of `I` may already be small.
- When the decoder needs `café`, the useful English signal is mostly in `coffee`. But it only has the combined vector $s$.

Soft attention [[1]](#ref-bahdanau) keeps every encoder vector and, at each French prediction step, computes a weighted sum over those English positions.

### Soft attention: use every encoder vector, not one summary

Yes: at each step we already have some French (from the reference in training, or from earlier predictions when generating), and we are about to **predict the next French word**. Soft attention is the part that, *before* that prediction, scores the English-side vectors and forms a context vector from them.

**Where is the “query”?** The previous French words are **not** the query themselves. You do not compare the string `J'aime` to the string `coffee`. The chain is:

```text
French so far  →  decoder hidden state  →  query vector q  →  score vs each English hᵢ
   (tokens)         (a vector)              (another vector)
```

1. Feed the French so far into the decoder; it keeps a [hidden state](/ai-glossary/hidden-state/) that summarizes that French history.
2. From that hidden state, a learned linear map produces a **query** $ q $, which is again just a vector of numbers (e.g. $ q = W_q\, s_{\mathrm{decoder}} $). After training, $ q $ tends to score highly against the encoder states that are useful for predicting the next French word.
3. Soft attention compares $ q $ to every encoder vector $ h_i $ (for `I`, `love`, `coffee`, …), turns those comparisons into weights, and builds a context vector for this step.
4. **Only then** does the decoder predict the next French word from its state plus that context (usually via a softmax over the French vocabulary).

Important: the query is **not** a guess of the next French word, and not an embedding of `café`. It is a vector used to score encoder states. The actual next-word prediction is a later step, after attention has produced a context vector.

So: French so far determines the decoder state; the decoder state is mapped to a query; the query selects English states by score; the next French token is predicted **after** that context vector is built.

**Concrete toy example** (made-up 3-D numbers while the real models use hundreds of dimensions):

Suppose French so far is `J'aime le`, and we are about to predict the next word (hopefully `café`). The decoder has already turned that French into a hidden-state vector, say:

$$
s_{\mathrm{decoder}} = (1.0,\ 0.0,\ 1.0)
$$

(The subscript “decoder” just marks this as the **decoder’s** hidden state, as opposed to an encoder $ h_i $.)

**Where do the query’s values come from?** From a **learned linear map** applied to that state, typically $ q = W_q\, s_{\mathrm{decoder}} $ (plus a bias). $ W_q $ is a matrix of trainable weights. After training, suppose

$$
W_q =
\begin{pmatrix}
0.9 & 0.0 & 0.0 \\
0.1 & 0.5 & 0.0 \\
0.0 & 0.0 & 0.8
\end{pmatrix}
\quad\Rightarrow\quad
q = W_q\, s_{\mathrm{decoder}} = (0.9,\ 0.1,\ 0.8).
$$

(In this blog the matrix entries are invented so the arithmetic stays small. In a real run, gradient descent on the translation loss sets $ W_q $ and the encoder that builds each $ h_i $, so that useful score patterns appear.)

Those three numbers in $ q $ are still not “the word café.” They are coordinates used only to **score** encoder vectors. Pretend the encoder produced:

$$
\begin{align*}
h_1\ (\texttt{I}) &= (0.8,\ 0.1,\ 0.0) \\
h_2\ (\texttt{love}) &= (0.2,\ 0.9,\ 0.1) \\
h_3\ (\texttt{coffee}) &= (0.7,\ 0.0,\ 0.9)
\end{align*}
$$

Score each English position with a plain **dot product** $ e_i = q^\top h_i $:

```text
e₁ = 0.9·0.8 + 0.1·0.1 + 0.8·0.0 = 0.73   (I)
e₂ = 0.9·0.2 + 0.1·0.9 + 0.8·0.1 = 0.35   (love)
e₃ = 0.9·0.7 + 0.1·0.0 + 0.8·0.9 = 1.35   (coffee)  ← highest
```

Softmax over $(0.73, 0.35, 1.35)$ then puts most weight on $h_3$. The query has a higher dot product with `coffee` than with `love` because training adjusted both $q$ and the $h_i$’s that way. This is not because we stored a coffee→café dictionary entry.

At a single decoder step, soft attention always scores **all** encoder vectors against $ q $, then puts more weight on whichever vectors reduce next-word loss.

That can still sound unclear. How does the model associate `coffee` with `café` when there is **no bilingual dictionary and no hard rule about positions**?

Here is what training actually does:

1. **Scores are learned.** Training on many English–French pairs adjusts the maps that build $ q $ and that score $ q $ against each $ h_i $. When the next target word is `café`, a well-trained model tends to produce a $ q $ that scores highly against the encoder vector for `coffee`. Bad weights → wrong next-word prediction → high loss → gradient descent updates those maps.
2. **Order need not match.** Softmax can put most weight on *any* English index, so French and English word order can differ. The $0.85$ on `coffee` below is an *illustrative* learned outcome, not something we typed into the code.

The next three quantities are:

1. **Score** $ e_i $: a learned similarity between query $ q $ and encoder state $ h_i $ (one number per English position). Common choices: a dot product, or a small MLP over $ [q; h_i] $. Scores only *rank* positions.
2. **Weights** $ \alpha_i $: softmax over those scores so they sum to 1. Each $\alpha_i$ is the coefficient on $ h_i $ in the context vector.
3. **Context vector** $ c = \sum_i \alpha_i h_i $: the weighted sum of English-side states the decoder will use for this step.

After training, a step that should produce `café` might look like:

```text
  I      love    coffee
  0.05   0.10    0.85     ← α₁, α₂, α₃ (sum to 1; learned, not hardcoded)

  c ≈ 0.05·h₁ + 0.10·h₂ + 0.85·h₃
```

That $ c $ is what the decoder uses for this step. On the next French word it forms a **new** query and a **new** weighted sum, so the coefficients can put mass on `I` when predicting `J'aime`, and on `coffee` when predicting `café`, even if those English words sit in a different order than the French.

<div class="mermaid">
flowchart LR
  subgraph Encoder["Encoder keeps every position"]
    E1["h₁: I"] --- E2["h₂: love"] --- E3["h₃: coffee"]
  end
  subgraph SoftAttention["Soft attention pipeline"]
    Sc["scores eᵢ<br/>match query to each hᵢ"]
    Sm["softmax → weights αᵢ"]
    Sc --> Sm
  end
  subgraph Decoder["Decoder step t"]
    Dt["query q"]
    Ct["context c = Σ αᵢ hᵢ"]
  end
  E1 & E2 & E3 --> Sc
  Dt --> Sc
  Sm --> Ct
  E1 & E2 & E3 --> Ct
</div>

The “soft” in soft attention means the weights are continuous probabilities from a softmax, not a hard one-hot pick of a single source position. The pattern **similarity scores → normalize → weighted sum** is the same pattern used later in self-attention inside Transformers.

## Soft attention in one formula

Given encoder states $ H = [h_1, \ldots, h_T] $ and a decoder query $ q_t $:

$$
e_{t,i} = \mathrm{score}(q_t, h_i), \quad
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}, \quad
c_t = \sum_i \alpha_{t,i}\, h_i
$$

Common score functions include a dot product $ q_t^\top h_i $ or a small [MLP](/ai-glossary/mlp/) (multi-layer perceptron: a few fully connected layers with a nonlinearity) over the concatenated pair $ [q_t; h_i] $. The context vector $ c_t $ is what the decoder uses as input for this step.

Self-attention in Transformers is the same pattern, applied *within* one sequence: every token is both a query source and a key/value source [[2]](#ref-vaswani).

## Simplified self-attention (no trainable weights)

Let's start with a tiny sentence and pretend each word is already an embedding vector:

> **rain softens dry soil**

Four tokens → four embedding vectors $ x_1, x_2, x_3, x_4 \in \mathbb{R}^d $. Fix one query position, say token 3 (“dry”), and ask: what weight should each token get in the weighted sum that becomes the output at position 3?

### Step A: raw scores

Take the query vector $ x_3 $ and measure similarity to every input via a dot product:

$$
\omega_i = x_3^\top x_i \quad (i = 1,\ldots,4)
$$


### Step B: turn scores into weights

$$
\alpha_i = \mathrm{softmax}(\omega)_i = \frac{\exp(\omega_i)}{\sum_j \exp(\omega_j)}
$$

The $ \alpha_i $ sum to 1. They are the attention distribution for position 3.

### Step C: form a context vector

$$
z_3 = \sum_i \alpha_i\, x_i
$$

$ z_3 $ is a weighted sum of all token embeddings, with larger coefficients on positions $i$ where $ x_3^\top x_i $ is larger.

<div class="mermaid">
flowchart TB
  X["inputs x₁…x₄"] --> DOT["ωᵢ = x₃ · xᵢ"]
  DOT --> SM["α = softmax(ω)"]
  SM --> MIX["z₃ = Σ αᵢ xᵢ"]
  X --> MIX
</div>

Repeat Steps A–C with $ x_1 $, $ x_2 $, and $ x_4 $ as the query and you get $ z_1, \ldots, z_4 $. There are still no trainable matrices: scores use the embeddings directly. The next section adds $ W_Q $, $ W_K $, and $ W_V $, which is what Transformer self-attention uses.

### All positions at once

With input matrix $ X \in \mathbb{R}^{T \times d} $ (rows = tokens):

$$
A = \mathrm{softmax}(X X^\top), \quad Z = A X
$$

$ A_{ij} $ is how much token $ i $ attends to token $ j $. Matrix sizes: $ X X^\top $ is $ T \times T $; $ Z $ is $ T \times d $.

```python
import torch
import torch.nn.functional as F

# T=4 tokens, d=3 dims. Toy embeddings for arithmetic, not from a real model.
X = torch.tensor([
    [0.4, 0.1, -0.2],  # rain
    [0.3, 0.5,  0.1],  # softens
    [0.1, 0.2,  0.6],  # dry
    [0.2, 0.0,  0.3],  # soil
], dtype=torch.float32)

scores = X @ X.T                 # (4, 4)
weights = F.softmax(scores, dim=-1)
Z = weights @ X                  # (4, 3) context vectors
```

## Adding trainable weights: queries, keys, values

The weight-free version uses the same vectors as queries, keys, *and* values. Real self-attention learns three linear maps so those three roles can differ:

- $Q$: the vectors used to score against keys
- $K$: the vectors that are scored
- $V$: the vectors that enter the weighted sum

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

with $ W_Q, W_K \in \mathbb{R}^{d \times d_k} $ and $ W_V \in \mathbb{R}^{d \times d_v} $. Often $ d_k = d_v $.

### Scaled scores and context

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

The $ \sqrt{d_k} $ scale keeps the softmax from saturating when $ d_k $ is large. Without it, large-magnitude dots push the softmax toward near one-hot weights and gradients become very small.

<div class="mermaid">
flowchart LR
  X[X] --> WQ["× WQ"] --> Q[Q]
  X --> WK["× WK"] --> K[K]
  X --> WV["× WV"] --> V[V]
  Q --> S["QKᵀ / √dₖ"]
  K --> S
  S --> SM[softmax]
  SM --> OUT["× V → Z"]
  V --> OUT
</div>

### Step-by-step for one query row

1. Project: $ q_i = x_i W_Q $, $ k_j = x_j W_K $, $ v_j = x_j W_V $.
2. Score: $ \omega_{ij} = q_i^\top k_j / \sqrt{d_k} $.
3. Normalize over key index $j$:
   $$\alpha_{ij} = \frac{\exp(\omega_{ij})}{\sum_{j'} \exp(\omega_{ij'})}$$
   (softmax along the row $\omega_{i,\cdot}$).
4. Weighted sum of values: $ z_i = \sum_j \alpha_{ij} v_j $.

Same four steps as soft attention. The new pieces are the three projections and the scale factor.

## A compact self-attention module

Wrap the math in a small `nn.Module`. This is the non-causal building block (bidirectional over the sequence).

```python
import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        # x: (batch, T, d_in)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        return weights @ V  # (batch, T, d_out)
```

Example usage:

```python
torch.manual_seed(0)
# X is (4, 3). unsqueeze(0) inserts a batch axis → (1, 4, 3)
# so it matches forward's expected layout (batch, T, d_in).
batch = X.unsqueeze(0)
attn = SelfAttention(d_in=3, d_out=2)  # instance of the class above
out = attn(batch)  # calls SelfAttention.forward → (1, 4, 2)
```

At this stage every token can attend to every other token, including later positions. That is fine for encoders. It is wrong for next-token language modeling.

## Causal attention: hide the future

**Causal (masked) attention** is self-attention with a constraint: query position $i$ may only use keys at positions $j \le i$. Future positions are blocked before the softmax, so they get weight 0 in the context vector.

That is required for [**autoregressive**](/ai-glossary/autoregressive/) language models. Without the mask, training would let position $i$ use tokens $i+1, i+2, \ldots$, which are not available when generating left to right.

| query \\ key | 1 | 2 | 3 | 4 |
|:------------:|:-:|:-:|:-:|:-:|
| **1** | ✓ | ✗ | ✗ | ✗ |
| **2** | ✓ | ✓ | ✗ | ✗ |
| **3** | ✓ | ✓ | ✓ | ✗ |
| **4** | ✓ | ✓ | ✓ | ✓ |

Implementation pattern: fill forbidden positions with $ -\infty $ (or a large negative), *then* softmax. Softmax of $ -\infty $ is 0, so future tokens contribute nothing to the context.

$$
\widetilde{\omega}_{ij} =
\begin{cases}
\omega_{ij} & j \le i \\
-\infty & j > i
\end{cases}
\quad\Rightarrow\quad
\alpha = \mathrm{softmax}(\widetilde{\omega})
$$

After masking, the upper triangle of the weight matrix is zero:

```text
α ≈
[[a11,  0,    0,    0  ],
 [a21, a22,   0,    0  ],
 [a31, a32,  a33,   0  ],
 [a41, a42,  a43,  a44]]
```

<div class="mermaid">
sequenceDiagram
  participant Q as Query pos i
  participant S as Scores
  participant M as Causal mask
  participant SM as Softmax
  participant V as Values
  Q->>S: qᵢ · kⱼ / √dₖ for all j
  S->>M: set j > i to -inf
  M->>SM: normalize remaining
  SM->>V: weighted sum of vⱼ, j ≤ i
</div>

## Dropout on attention weights

Even with a causal mask, the attention matrix can overfit by fitting brittle token-to-token coefficients. A standard regularizer is **dropout on the attention weights** (after softmax, before multiplying by $ V $). During training, randomly zero some $ \alpha_{ij} $ and rescale the rest; at eval time, leave weights intact.

<div class="mermaid">
flowchart LR
  SM[softmax weights] --> DO[dropout]
  DO --> MV["× V"]
</div>

Dropout here is *not* applied to the embeddings or to Q/K/V projections in this module; it zeros some of the mixing coefficients. Typical rates in GPT-style models are modest (e.g. 0.1).

Order of operations that matters:

1. Compute scaled scores $ Q K^\top / \sqrt{d_k} $
2. Apply causal mask ($ -\infty $ on the future)
3. Softmax over keys
4. Dropout on the weight matrix
5. Multiply by $ V $

Swap (2) and (3) and you no longer have a proper distribution over the *allowed* positions.

## Causal attention: compact code

The module below puts Q/K/V, the causal mask, and dropout in one `forward` function. The mask is a fixed boolean matrix (upper triangle = future positions). It is stored with `register_buffer` so PyTorch keeps it on the same device as the module when you call `.to(device)` or `.cuda()`, but does **not** treat it as a trainable weight (no gradients, not in `parameters()`).

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.1, bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # True above the diagonal = future key positions → filled with -inf later
        mask = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        )
        self.register_buffer("mask", mask.bool())  # fixed tensor, not a learned param

    def forward(self, x):
        # x: (B, T, d_in); T ≤ context_length
        B, T, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ V
```

Quick check on tensor sizes:

```python
torch.manual_seed(1)
x = torch.randn(2, 4, 3)  # batch=2, T=4, d=3
m = CausalSelfAttention(d_in=3, d_out=2, context_length=4, dropout=0.0)
y = m(x)
assert y.shape == (2, 4, 2)
```

With `dropout=0.0` and a fixed seed you can inspect `weights` and confirm the upper triangle is exactly zero.

## From one head to many

A single attention head learns one set of projections and one attention pattern. **Multi-head attention** runs several heads in parallel, each with its own $ W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} $, then concatenates their outputs and applies an output matrix. Different heads can learn different score patterns (for example local adjacency versus longer-range links).

<div class="mermaid">
flowchart TB
  X[X] --> H1[Head 1: Attn]
  X --> H2[Head 2: Attn]
  X --> H3[Head 3: Attn]
  H1 --> CAT[concat]
  H2 --> CAT
  H3 --> CAT
  CAT --> WO["× WO"]
  WO --> Z[Z]
</div>

### Approach 1: stack single-head modules

Build `n_heads` separate `CausalSelfAttention` modules. If the multi-head block should output width `d_out`, give each head width `head_dim = d_out // n_heads`. Run every head on the same input `x`. Each head returns a tensor of shape `(B, T, head_dim)`. Concatenate those outputs along the last dimension to get `(B, T, d_out)`, then multiply by a learned matrix `out_proj` ($W_O$):

```python
class MultiHeadCausalAttentionStacked(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout=0.1):
        super().__init__()
        assert d_out % n_heads == 0
        head_dim = d_out // n_heads
        self.heads = nn.ModuleList([
            CausalSelfAttention(d_in, head_dim, context_length, dropout)
            for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        # each head → (B, T, head_dim); concat → (B, T, d_out)
        h = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.out_proj(h)
```

Easy to read. A bit wasteful, because each head repeats the same mask/dropout structure and does separate small [**matmuls**](/ai-glossary/matmul/) (matrix multiplications, e.g. $QK^\top$ or weights times $V$).

### Approach 2: one big matrix multiply, then split

Instead of `n_heads` small modules, multiply once with wide $W_Q$, $W_K$, $W_V$ that output width $n_{\mathrm{heads}} \cdot d_k$, reshape that width into heads, run attention on all heads in one batched matmul, concatenate heads again, then multiply by $W_O$. Same math as Approach 1, usually faster on a GPU.

```python
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=bias)
        self.W_k = nn.Linear(d_in, d_out, bias=bias)
        self.W_v = nn.Linear(d_in, d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(
            torch.ones(context_length, context_length), diagonal=1
        )
        self.register_buffer("mask", mask.bool())

    def _split_heads(self, t):
        # (B, T, d_out) → (B, n_heads, T, head_dim)
        B, T, _ = t.shape
        t = t.view(B, T, self.n_heads, self.head_dim)
        return t.transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        ctx = weights @ V                         # (B, heads, T, head_dim)
        ctx = ctx.transpose(1, 2).contiguous()    # (B, T, heads, head_dim)
        ctx = ctx.view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(ctx)
```

<div class="mermaid">
flowchart LR
  subgraph per_head["per head h"]
    Qh[Qh] --> Sh["scores"]
    Kh[Kh] --> Sh
    Sh --> Mh[mask + softmax + drop]
    Mh --> Zh["Zh = · Vh"]
    Vh[Vh] --> Zh
  end
  Zh --> Merge["concat heads → WO"]
</div>

Both approaches implement:

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)\, W_O
$$

$$
\mathrm{head}_h = \mathrm{CausalAttn}(X W_Q^{(h)}, X W_K^{(h)}, X W_V^{(h)})
$$

## Putting the pieces together

<div class="attn-ladder" aria-label="Ladder from soft attention to multi-head">
  <ol class="attn-ladder-row">
    <li><strong>1</strong> Soft attention<span>encoder–decoder</span></li>
    <li><strong>2</strong> Weight-free self-attn<span>softmax XXᵀ · X</span></li>
    <li><strong>3</strong> Trainable Q, K, V<span>+ √d<sub>k</sub> scale</span></li>
    <li><strong>4</strong> SelfAttention<span>module</span></li>
  </ol>
  <div class="attn-ladder-wrap" aria-hidden="true">
    <svg width="100%" height="100%" overflow="visible">
      <g class="attn-ladder-wrap-desktop">
        <line x1="87.5%" y1="0" x2="87.5%" y2="50%" />
        <line x1="87.5%" y1="50%" x2="12.5%" y2="50%" />
        <line x1="12.5%" y1="50%" x2="12.5%" y2="100%" />
      </g>
      <g class="attn-ladder-wrap-mobile">
        <line x1="75%" y1="0" x2="75%" y2="50%" />
        <line x1="75%" y1="50%" x2="25%" y2="50%" />
        <line x1="25%" y1="50%" x2="25%" y2="100%" />
      </g>
    </svg>
    <span class="attn-ladder-wrap-arrow"></span>
  </div>
  <ol class="attn-ladder-row" start="5">
    <li><strong>5</strong> Causal mask<span>block future keys</span></li>
    <li><strong>6</strong> Dropout<span>on α</span></li>
    <li><strong>7</strong> CausalSelfAttention<span>masked + dropout</span></li>
    <li><strong>8</strong> Multi-head<span>stack or split</span></li>
  </ol>
</div>

| Stage | What you gain |
|-------|----------------|
| Soft attention | Weighted sum over a set of vectors, with learned coefficients |
| Weight-free self-attn | Same computation inside one sequence; no new params |
| Q, K, V + scale | Separate projections for scoring and for the sum; stable softmax |
| Causal mask | No access to future tokens; usable for next-token prediction |
| Dropout on weights | Regularization of the attention coefficients |
| Multi-head | Several attention patterns in parallel |

## Checklist

- Attention is score, normalize, weighted sum.
- Self-attention applies that computation among tokens of the *same* sequence.
- Trainable Q/K/V (and scaling) make the projections learnable and keep softmax numerically stable.
- Causal masks enforce the autoregressive rule; dropout regularizes the mixing weights.
- Multi-head attention is several causal heads concatenated, then multiplied by $ W_O $.

From here, the next pieces of a decoder-only Transformer are residual connections, layer norm, and the position-wise feed-forward block. Attention combines information across tokens. Those other parts transform each token’s vector independently.

## References

1. <span id="ref-bahdanau"></span>Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural Machine Translation by Jointly Learning to Align and Translate.” In *International Conference on Learning Representations (ICLR)*, 2015. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473){:target="_blank" rel="noopener noreferrer"}.  
   Read in [BetterReads](https://www.betterreads.dev/reader?bookId=bahdanau_nmt_align_translate&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
2. <span id="ref-vaswani"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. “Attention Is All You Need.” In *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762){:target="_blank" rel="noopener noreferrer"}.  
   Read in [BetterReads](https://www.betterreads.dev/reader?bookId=vaswani_attention_is_all_you_need&type=pdf&preview=1){:target="_blank" rel="noopener noreferrer"}
3. Zhao Lü. [*Recurrent Neural Network (RNN): From Theory to Translation*](/2025/06/10/rnn-machine-translation.html){:target="_blank" rel="noopener noreferrer"}. Encoder/decoder RNNs before attention.
