---
title: Quantile
synonyms:
  - Percentile
  - Fractile
keywords: quantile, percentile, quartile, critical value, statistics
related:
  - Median
  - Cumulative distribution function
  - Order statistic
  - Interquartile range
  - Confidence interval
---
A **quantile** of order $p\in(0,1)$ is a value $q_p$ such that $F(q_p)\ge p$ (and $P(X\ge q_p)\ge 1-p$) — roughly, fraction $p$ of the distribution lies at or below $q_p$. Special cases: [median](/cs-glossary/median/) ($p=0.5$), quartiles ($p=0.25,0.75$).

Sample quantiles are read from [order statistics](/cs-glossary/order-statistic/) or the empirical cdf. Critical values for tests are quantiles of null reference distributions (e.g. $t_{n-2,1-\alpha/2}$).
