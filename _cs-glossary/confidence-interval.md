---
title: Confidence interval
synonyms:
  - CI
  - Interval estimate
  - Interval estimation
keywords: confidence interval, CI, interval estimation, coverage, statistics glossary, probability
related:
  - Standard error
  - T-test
  - Point estimation
  - Hypothesis testing
  - Quantile
---
A **confidence interval** (CI) is an interval estimator for $\theta$. A level $1-\alpha$ CI satisfies $P_\theta(\text{interval covers }\theta)=1-\alpha$ under the model (coverage of the *procedure*, not a posterior probability for one realised interval).

Simple OLS slope: $\hat\beta\pm t_{n-2,1-\alpha/2}\,\mathrm{s.e.}(\hat\beta)$. The CI inverts the [$t$-test](/cs-glossary/t-test/): reject $H_0:\beta=c$ iff $c$ is outside the interval.
