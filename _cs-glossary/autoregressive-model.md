---
title: Autoregressive model
synonyms:
  - AR model
  - AR(p)
  - Autoregression
keywords: autoregressive model, AR(p), AR, PACF, time series, statistics glossary
related:
  - Moving average model
  - ARMA
  - ARIMA
  - Partial autocorrelation
  - Stationarity
  - Lag
---
An **autoregressive** model of order $p$, AR($p$), regresses $Y_t$ on its own lags:

$$
Y_t = b_0 + \theta_1 Y_{t-1}+\cdots+\theta_p Y_{t-p}+w_t,
$$

with [$w_t$ white noise](/cs-glossary/white-noise/). Identification: [ACF](/cs-glossary/autocorrelation-function/) decays; [PACF](/cs-glossary/partial-autocorrelation/) spikes cut off at lag $p$. Needs [stationarity](/cs-glossary/stationarity/) (roots outside unit circle). Building block of [ARMA](/cs-glossary/arma/)/[ARIMA](/cs-glossary/arima/).
