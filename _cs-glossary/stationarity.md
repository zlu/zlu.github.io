---
title: Stationarity
synonyms:
  - Weak stationarity
  - Covariance stationarity
  - Strict stationarity
keywords: stationarity, weak stationarity, unit root, ARMA, time series, statistics glossary
related:
  - Differencing
  - ARMA
  - ARIMA
  - Trend
  - Seasonality
  - White noise
---
A series is **(weakly) stationary** if mean and variance are constant over time and $\mathrm{Cov}(Y_t,Y_{t-k})$ depends only on lag $k$ — and typically no [seasonality](/cs-glossary/seasonality/).

[ARMA](/cs-glossary/arma/) assumes stationarity. Nonstationary levels: [difference](/cs-glossary/differencing/) ($Z_t=Y_t-Y_{t-1}$) or detrend before modelling → path to [ARIMA](/cs-glossary/arima/).
