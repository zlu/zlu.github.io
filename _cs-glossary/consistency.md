---
title: Consistency
synonyms:
  - Consistent estimator
  - Weak consistency
keywords: consistency, consistent estimator, convergence in probability, LLN, statistics glossary, probability
related:
  - Convergence in probability
  - Law of large numbers
  - Asymptotic normality
  - Bias
  - Estimator
---
An [estimator](/cs-glossary/estimator/) sequence $W_n$ is **consistent** for $\theta$ if $W_n\xrightarrow{p}\theta$ ([convergence in probability](/cs-glossary/convergence-in-probability/)). Enough conditions: bias $\to 0$ and variance $\to 0$.

The [LLN](/cs-glossary/law-of-large-numbers/) makes $\bar X_n$ consistent for $\mu$. Consistency is asymptotic — it does not guarantee small error at small $n$. **CAN** estimators are consistent and [asymptotically normal](/cs-glossary/asymptotic-normality/).
