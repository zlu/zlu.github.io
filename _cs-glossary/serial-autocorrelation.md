---
title: Serial autocorrelation
synonyms:
  - Serial correlation of residuals
  - Temporal autocorrelation
keywords: serial autocorrelation, residual autocorrelation, Durbin-Watson, time series regression, statistics glossary
related:
  - Autocorrelation
  - Durbin–Watson
  - Time series
  - Autocorrelation function
  - Gauss–Markov theorem
---
**Serial autocorrelation** is [autocorrelation](/cs-glossary/autocorrelation/) along time — especially of regression residuals $e_t$. Positive serial correlation: consecutive residuals tend to share sign; negative: they tend to flip.

Violates Gauss–Markov independence ⇒ OLS inefficient and classical SEs too small. Diagnose with [Durbin–Watson](/cs-glossary/durbin-watson/) $d$ and residual [ACF](/cs-glossary/autocorrelation-function/).
