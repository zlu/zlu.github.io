---
title: Asymptotic normality
synonyms:
  - CAN
  - Asymptotically normal
  - Consistent and asymptotically normal
keywords: asymptotic normality, CAN estimator, Wald, MLE, statistics
related:
  - Central limit theorem
  - Consistency
  - Maximum likelihood
  - Standard error
  - Confidence interval
---
An [estimator](/cs-glossary/estimator/) $W_n$ is **asymptotically normal** if $\sqrt{n}(W_n-\theta)/\sigma_\theta\xrightarrow{d}N(0,1)$ for some asymptotic variance $\sigma_\theta^2$. Together with [consistency](/cs-glossary/consistency/), this is the **CAN** property.

The [CLT](/cs-glossary/central-limit-theorem/) gives asymptotic normality of $\bar X$. Regular [MLEs](/cs-glossary/maximum-likelihood/) are typically CAN, enabling Wald intervals and tests from an estimated $\widehat{\mathrm{s.e.}}$.
