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
  - Residual standard error
  - Null hypothesis
  - P-value
  - F-test
---
A **$t$-test** compares an estimate to a null value using a $t$ reference distribution. For simple OLS with Normal iid errors, under $H_0:\beta=c$ (or $H_0:\alpha=c$),

$$
t_{\hat\beta}=\frac{\hat\beta-c}{\mathrm{s.e.}(\hat\beta)},\qquad
t_{\hat\alpha}=\frac{\hat\alpha-c}{\mathrm{s.e.}(\hat\alpha)},
\quad\sim t_{n-2},
$$

with $\mathrm{s.e.}(\hat\beta)=s/\sqrt{S_{xx}}$ and $\mathrm{s.e.}(\hat\alpha)=s\sqrt{1/n+\bar x^2/S_{xx}}$ ([RSE](/cs-glossary/residual-standard-error/) $s$). Reject two-sided $H_0$ at level $p$ if $\lvert t\rvert>t_{n-2,\,p/2}$. Typical significance probe: $c=0$. Equivalent to checking whether $c$ lies outside the [confidence interval](/cs-glossary/confidence-interval/). In simple regression, $F=t^2$ for the slope vs the joint [$F$-test](/cs-glossary/f-test/).
