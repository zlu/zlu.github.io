---
title: Confidence interval
synonyms:
  - CI
  - Interval estimate
  - Interval estimation
keywords: confidence interval, CI, interval estimation, coverage, OLS, statistics glossary, probability
related:
  - Standard error
  - T-test
  - Point estimation
  - Hypothesis testing
  - Residual standard error
  - Quantile
---
A **confidence interval** (CI) is an interval estimator for $\theta$. A level $1-\alpha$ CI satisfies $P_\theta(\text{interval covers }\theta)=1-\alpha$ under the model (coverage of the *procedure*, not a posterior probability for one realised interval).

For simple OLS with Normal errors, invert the [$t$-test](/cs-glossary/t-test/):

$$
\beta:\ \hat\beta\pm t_{n-2,\,1-p/2}\,\mathrm{s.e.}(\hat\beta),\qquad
\alpha:\ \hat\alpha\pm t_{n-2,\,1-p/2}\,\mathrm{s.e.}(\hat\alpha),
$$

and for the mean response at $x$,

$$
\hat y(x)\pm t_{n-2,\,1-p/2}\,s\sqrt{\frac{1}{n}+\frac{(x-\bar x)^2}{S_{xx}}}.
$$

Reject $H_0:\beta=c$ iff $c$ is outside the slope interval.
