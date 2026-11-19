---
title: Expectation
synonyms:
  - Expected value
  - Mean
  - Population mean
keywords: expectation, expected value, E[X], mean, statistics glossary
related:
  - Variance
  - Law of large numbers
  - Sample mean
  - Linearity of expectation
---
**Expectation** (expected value, mean) of a [random variable](/cs-glossary/random-variable/) $X$ is $E[X]=\sum x f(x)$ (discrete) or $\int x f(x)\,dx$ (continuous). Linearity always holds: $E[ag(X)+bh(Y)]=aE[g(X)]+bE[h(Y)]$ whether or not $X,Y$ are independent.

$E[X]$ is a property of the distribution, not of one sample. The [sample mean](/cs-glossary/sample-mean/) estimates it; the [LLN](/cs-glossary/law-of-large-numbers/) says $\bar X_n\to E[X]$ in probability.
