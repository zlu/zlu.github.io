---
title: Dependent Variable
synonyms: [Response Variable, Output Variable, Target Variable]
related: [Independent Variable, Regression, Supervised Learning, Loss Function]
---
A **dependent variable** is the quantity a model or experiment aims to explain or predict. Its value is treated as a function of one or more [independent variables](/ai-glossary/independent-variable/) (features). In [supervised learning](/ai-glossary/supervised-learning/), it is the labeled target: the class, score, next token, or continuous response.

### Why it matters
- Defines what the [loss function](/ai-glossary/loss-function/) compares predictions against
- Choice of target (raw label vs transformed) shapes difficulty and metrics
- Leakage into features can make a “dependent” variable look trivially predictable

### How it shows up
In classical stats you often write $y = f(X) + \epsilon$. In ML pipelines the same idea is `label` / `target` columns. For language models, the dependent variable at each step is typically the next [token](/ai-glossary/token/) (or a masked token in bidirectional models).

Keep naming precise in write-ups: “dependent variable,” “response,” and “target” are interchangeable; “output” can confuse people with model logits versus ground truth.
