---
title: ARIMA
synonyms:
  - ARIMA(p,d,q)
  - Autoregressive integrated moving average
keywords: ARIMA, ARIMA(p,d,q), differencing, integrated, forecasting, statistics glossary
related:
  - ARMA
  - Differencing
  - Stationarity
  - Autoregressive model
  - Forecasting
---
**ARIMA($p,d,q$)** is [ARMA](/cs-glossary/arma/) applied after $d$-th [differencing](/cs-glossary/differencing/) to achieve [stationarity](/cs-glossary/stationarity/):

$$
Z_t=\nabla^d Y_t=(1-L)^d Y_t,
$$

then ARMA($p,q$) on $Z_t$. The “I” = integrated. Standard Box–Jenkins forecasting workhorse when levels have trends but differences look stationary.
