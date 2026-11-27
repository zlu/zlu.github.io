---
title: Durbin–Watson
synonyms:
  - DW statistic
  - Durbin-Watson
  - d-statistic
keywords: Durbin-Watson, serial correlation, residual diagnostics, dL dU, time series, statistics glossary
related:
  - Serial autocorrelation
  - Residual
  - Autocorrelation
  - Autocorrelation function
  - Positive serial correlation
  - Negative serial correlation
  - Time series
  - Standard error
---
The **Durbin–Watson** statistic tests first-order [serial correlation](/cs-glossary/serial-autocorrelation/) in regression residuals:

$$
d=\frac{\sum_{t=2}^{n}(e_t-e_{t-1})^2}{\sum_{t=1}^{n}e_t^2}\in(0,4).
$$

| $d$ | Reading |
|---|---|
| $\approx 2$ | little AR(1) correlation |
| $\ll 2$ (esp. $<1$) | [positive](/cs-glossary/positive-serial-correlation/) serial correlation |
| $\gg 2$ | [negative](/cs-glossary/negative-serial-correlation/) serial correlation |

Formal test: compare $d$ to tabulated bounds $(d_L,d_U)$ for given $n$, $k$, and $\alpha$. If $d$ flags dependence, classical SEs and $t$/$p$ mislead — use AR/ARIMA-style models or HAC SEs.
