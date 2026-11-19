---
title: Standard error
synonyms:
  - SE
  - s.e.
keywords: standard error, SE, s.e., sampling variability, statistics glossary, probability
related:
  - Sampling distribution
  - T-test
  - Confidence interval
  - Standard deviation
  - Residual sum of squares
---
A **standard error** (SE) is an estimate of the standard deviation of an [estimator](/cs-glossary/estimator/)'s [sampling distribution](/cs-glossary/sampling-distribution/). For the simple OLS slope,

$$
\mathrm{s.e.}(\hat\beta)=\frac{s}{\sqrt{S_{xx}}},\quad s=\sqrt{\mathrm{RSS}/(n-2)}.
$$

SEs drive [$t$-statistics](/cs-glossary/t-test/) and [confidence intervals](/cs-glossary/confidence-interval/). Wrong assumptions ([hetero](/cs-glossary/heteroscedasticity/), dependence) ⇒ wrong SEs even if $\hat\beta$ looks fine.
