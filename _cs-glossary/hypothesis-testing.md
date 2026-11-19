---
title: Hypothesis testing
synonyms:
  - Statistical test
  - Significance test
keywords: hypothesis testing, significance test, type I type II, p-value, statistics glossary, probability
related:
  - Null hypothesis
  - P-value
  - T-test
  - Confidence interval
  - Statistical inference
---
**Hypothesis testing** decides whether data are compatible with a claim about $\theta$. Specify [null](/cs-glossary/null-hypothesis/) $H_0$ and alternative $H_1$, choose a test statistic, and reject $H_0$ when the statistic falls in a critical region (or when the [$p$-value](/cs-glossary/p-value/) $<\alpha$).

Type I error: reject a true $H_0$ (rate $\alpha$). Type II: fail to reject a false $H_0$. Power $=1-$ Type II. Regression slope tests are usually two-sided $H_0:\beta=0$ via a [$t$-test](/cs-glossary/t-test/).
