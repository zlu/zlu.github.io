---
title: ARMA
synonyms:
  - ARMA(p,q)
  - Autoregressive moving average
keywords: ARMA, ARMA(p,q), Box-Jenkins, ACF PACF, time series, statistics glossary
related:
  - Autoregressive model
  - Moving average model
  - ARIMA
  - Stationarity
  - White noise
  - Autocorrelation function
---
An **ARMA($p,q$)** model combines [AR($p$)](/cs-glossary/autoregressive-model/) and [MA($q$)](/cs-glossary/moving-average-model/) on a [stationary](/cs-glossary/stationarity/) series with [white-noise](/cs-glossary/white-noise/) errors.

Order selection from [ACF](/cs-glossary/autocorrelation-function/)/[PACF](/cs-glossary/partial-autocorrelation/): ACF tails & PACF cuts ⇒ AR; PACF tails & ACF cuts ⇒ MA; both tail ⇒ ARMA. No seasonality and constant mean/variance assumed before fitting. Nonstationary series need [ARIMA](/cs-glossary/arima/) (differencing).
