---
title: Hypothesis testing
synonyms:
  - Statistical test
  - Significance test
keywords: hypothesis testing, significance test, type I type II, p-value, critical region, statistics glossary, probability
related:
  - Hypothesis
  - Null hypothesis
  - Alternative hypothesis
  - Test statistic
  - Critical region
  - P-value
  - Type I error
  - Type II error
  - Significance level
  - Power of a test
  - T-test
  - Confidence interval
---
**Hypothesis testing** decides whether data are compatible with a claim about $\theta$. Partition the parameter space into complementary $H_0:\theta\in\Omega_0$ and $H_1:\theta\in\Omega_0^c$, choose a [test statistic](/cs-glossary/test-statistic/) $T(X)$ with known null distribution, and reject $H_0$ when $T$ falls in the [critical region](/cs-glossary/critical-region/) $R$ — equivalently when the [$p$-value](/cs-glossary/p-value/) $\le\alpha$.

- [Type I error](/cs-glossary/type-i-error/): reject true $H_0$ (controlled by [size](/cs-glossary/significance-level/) $\alpha$)
- [Type II error](/cs-glossary/type-ii-error/): keep false $H_0$; [power](/cs-glossary/power-of-a-test/) $=1-\beta$
- Regression slopes: usually two-sided $H_0:\beta=0$ via a [$t$-test](/cs-glossary/t-test/)
