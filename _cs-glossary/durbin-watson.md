---
title: Durbin–Watson
synonyms:
  - DW statistic
  - Durbin-Watson
  - d-statistic
keywords: Durbin-Watson, serial correlation, residual diagnostics, time series, statistics
related:
  - Residual
  - Autocorrelation
  - Heteroscedasticity
  - Standard error
  - Time series
---
The **Durbin–Watson** statistic $d$ tests first-order serial correlation in regression residuals. Rough guide: $d\approx 2$ suggests little AR(1) correlation; $d$ near $0$ suggests positive serial correlation; $d$ near $4$ suggests negative.

Used heavily in time-series / trip-generation diagnostics. If $d$ flags dependence, classical SEs and [$t$](/cs-glossary/t-test/) tests mislead — consider AR/ARIMA-style fixes or HAC SEs rather than ignoring the signal.
