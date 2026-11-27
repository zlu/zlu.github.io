---
title: Autocorrelation
synonyms:
  - Serial correlation
  - ACF dependence
keywords: autocorrelation, serial correlation, ACF, PACF, Durbin-Watson, time series, statistics glossary
related:
  - Serial autocorrelation
  - Autocorrelation function
  - Partial autocorrelation
  - Durbin–Watson
  - Spatial autocorrelation
  - Covariance
  - Time series
  - Gauss–Markov theorem
  - White noise
---
**Autocorrelation** is correlation of a series (or of residuals) with its own [lags](/cs-glossary/lag/): $\mathrm{Corr}(Y_t,Y_{t-k})\neq 0$. In regression errors it violates the uncorrelated-errors part of [Gauss–Markov](/cs-glossary/gauss-markov-theorem/).

- Temporal: [serial autocorrelation](/cs-glossary/serial-autocorrelation/), [ACF](/cs-glossary/autocorrelation-function/), [PACF](/cs-glossary/partial-autocorrelation/), [Durbin–Watson](/cs-glossary/durbin-watson/)
- Spatial: [spatial autocorrelation](/cs-glossary/spatial-autocorrelation/), [Moran's I](/cs-glossary/morans-i/)

Consequences: inefficient OLS and invalid classical inference. Remedies: lagged / ARIMA models, spatial lag/error models, HAC SEs.
