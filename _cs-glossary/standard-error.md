---
title: Standard error
synonyms:
  - SE
  - s.e.
keywords: standard error, SE, s.e., sampling variability, OLS, statistics glossary, probability
related:
  - Sampling distribution
  - Residual standard error
  - T-test
  - Confidence interval
  - Standard deviation
  - Residual sum of squares
---
A **standard error** (SE) is an estimate of the standard deviation of an [estimator](/cs-glossary/estimator/)'s [sampling distribution](/cs-glossary/sampling-distribution/): $\mathrm{s.e.}(\hat\theta)=\sqrt{\widehat{\mathrm{Var}}(\hat\theta)}$. For the simple OLS slope and intercept, plug [RSE](/cs-glossary/residual-standard-error/) $s=\sqrt{\mathrm{RSS}/(n-2)}$ into

$$
\mathrm{s.e.}(\hat\beta)=\frac{s}{\sqrt{S_{xx}}},\qquad
\mathrm{s.e.}(\hat\alpha)=s\sqrt{\frac{1}{n}+\frac{\bar x^2}{S_{xx}}}.
$$

Population variances are $\sigma_\beta^2=\sigma^2/S_{xx}$ and $\sigma_\alpha^2=(\sigma^2/nS_{xx})\sum x_i^2$. SEs drive [$t$-statistics](/cs-glossary/t-test/) and [confidence intervals](/cs-glossary/confidence-interval/). Wrong assumptions ([hetero](/cs-glossary/heteroscedasticity/), dependence) ⇒ wrong SEs even if $\hat\beta$ looks fine.
