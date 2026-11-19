---
title: Autocorrelation
synonyms:
  - Serial correlation
  - ACF dependence
keywords: autocorrelation, serial correlation, ACF, Durbin-Watson, time series statistics
related:
  - Durbin–Watson
  - Covariance
  - Time series
  - Gauss–Markov theorem
---
**Autocorrelation** (serial correlation) is correlation of a series with its own lags: $\mathrm{Corr}(\varepsilon_t,\varepsilon_{t-k})\neq 0$. In regression errors it violates the uncorrelated-errors part of [Gauss–Markov](/cs-glossary/gauss-markov-theorem/).

Detect with residual plots, ACF/PACF, or [Durbin–Watson](/cs-glossary/durbin-watson/). Consequences: inefficient OLS and invalid classical inference. Remedies include lagged models, ARIMA errors, and HAC standard errors.
