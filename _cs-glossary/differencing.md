---
title: Differencing
synonyms:
  - First difference
  - Integration order
  - ΔY
keywords: differencing, first difference, integrated series, ARIMA, stationarity, statistics glossary
related:
  - Stationarity
  - ARIMA
  - Trend
  - Time series
---
**Differencing** removes a stochastic/[linear trend](/cs-glossary/trend/) by working with changes:

$$
Z_t=Y_t-Y_{t-1}.
$$

If $Y_t=b_0+b_1 t+u_t$ with iid $u_t$, then $E[Z_t]=b_1$ and $\mathrm{Var}(Z_t)=2\sigma_u^2$ under independence of consecutive shocks. Order of integration $d$ is how many differences yield a [stationary](/cs-glossary/stationarity/) series ([ARIMA](/cs-glossary/arima/)’s $d$).
