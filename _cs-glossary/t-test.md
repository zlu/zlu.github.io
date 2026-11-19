---
title: T-test
synonyms:
  - Student's t-test
  - t statistic
  - t-statistic
keywords: t-test, t statistic, hypothesis test, OLS inference, statistics glossary, probability
related:
  - Hypothesis testing
  - Confidence interval
  - Standard error
  - Null hypothesis
  - P-value
---
A **$t$-test** compares an estimate to a null value using a $t$ reference distribution. For simple OLS with Normal iid errors, under $H_0:\beta=c$,

$$
t=\frac{\hat\beta-c}{\mathrm{s.e.}(\hat\beta)}\sim t_{n-2}.
$$

Reject two-sided $H_0$ at level $\alpha$ if $|t|>t_{n-2,1-\alpha/2}$. Equivalent to checking whether $c$ lies outside the [confidence interval](/cs-glossary/confidence-interval/). Large-sample $z$ tests replace $t$ when the [CLT](/cs-glossary/central-limit-theorem/) applies and df is large.
