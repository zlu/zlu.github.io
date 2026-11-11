---
title: Independent Variable
synonyms: [Predictor, Feature, Input Variable, Explanatory Variable]
related: [Dependent Variable, Regression, Supervised Learning, Feature Engineering]
---
An **independent variable** is an input used to explain or predict a [dependent variable](/ai-glossary/dependent-variable/). In ML it is usually called a **feature** or **predictor**: tabular columns, tokens, pixels, or embeddings that the model conditions on.

### Why it matters
- Feature choice and quality dominate classical ML performance
- Collinear or leaked features distort interpretation and metrics
- In LLMs, “features” are largely learned representations; the explicit inputs are tokens and context

### How it shows up
Regression write-ups still say $X$ for independents and $y$ for the target. [Feature engineering](/ai-glossary/feature-engineering/) builds $X$; [supervised learning](/ai-glossary/supervised-learning/) fits $f(X) \approx y$. Watch for multicollinearity when interpreting coefficients, and for train–serve skew when production features drift from training.

In generative settings the conditioning context (prompt, retrieved chunks, tool results) plays the same role: inputs the model conditions on to produce the next outputs.
