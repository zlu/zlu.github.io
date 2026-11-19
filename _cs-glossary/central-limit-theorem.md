---
title: Central limit theorem
synonyms:
  - CLT
  - Classical CLT
keywords: central limit theorem, CLT, asymptotic normal, sample mean, statistics glossary, probability
related:
  - Law of large numbers
  - Normal distribution
  - Asymptotic normality
  - Sampling distribution
  - Z-score
---
The **central limit theorem** (CLT): for iid $X_i$ with mean $\mu$ and variance $\sigma^2\in(0,\infty)$,

$$
\frac{\sqrt{n}(\bar X_n-\mu)}{\sigma}\xrightarrow{d}N(0,1).
$$

Averages become approximately [Normal](/cs-glossary/normal-distribution/) for large $n$, regardless of the parent shape (with finite variance). Foundation for large-sample [$z$/$t$](/cs-glossary/t-test/) intervals and [asymptotic normality](/cs-glossary/asymptotic-normality/) of estimators.
