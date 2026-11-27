---
title: Partial autocorrelation
synonyms:
  - PACF
  - Partial ACF
keywords: partial autocorrelation, PACF, AR order, time series, statistics glossary
related:
  - Autocorrelation function
  - Autoregressive model
  - Lag
  - ARMA
  - Time series
---
**Partial autocorrelation** at lag $k$ is the correlation between $Y_t$ and $Y_{t-k}$ after removing intervening lags $Y_{t-1},\ldots,Y_{t-k+1}$:

$$
\mathrm{PACF}(k)=\mathrm{Corr}(Y_t,Y_{t-k}\mid Y_{t-1},\ldots,Y_{t-k+1}).
$$

Computed as the last coefficient $\theta_k$ in the regression $Y_t=b_0+\theta_1 Y_{t-1}+\cdots+\theta_k Y_{t-k}+w_t$. PACF cuts off after lag $p$ for an [AR($p$)](/cs-glossary/autoregressive-model/) process.
