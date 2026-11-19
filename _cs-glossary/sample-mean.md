---
title: Sample mean
synonyms:
  - Arithmetic mean
  - Empirical mean
  - X-bar
keywords: sample mean, x-bar, average, LLN CLT, statistics
related:
  - Expectation
  - Law of large numbers
  - Central limit theorem
  - Sample variance
  - Estimator
---
The **sample mean** $\bar X=n^{-1}\sum_{i=1}^n X_i$ estimates a population mean $\mu$. For [iid](/cs-glossary/iid/) data with mean $\mu$, $E[\bar X]=\mu$ ([unbiased](/cs-glossary/unbiased-estimator/)) and $\mathrm{Var}(\bar X)=\sigma^2/n$.

[LLN](/cs-glossary/law-of-large-numbers/): $\bar X\xrightarrow{p}\mu$. [CLT](/cs-glossary/central-limit-theorem/): $\sqrt{n}(\bar X-\mu)/\sigma\xrightarrow{d}N(0,1)$. It is MoM/MLE for many one-parameter families (Bernoulli $p$, Poisson $\lambda$, Normal $\mu$).
