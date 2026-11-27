---
title: Moving average model
synonyms:
  - MA model
  - MA(q)
keywords: moving average model, MA(q), MA, ACF, time series, statistics glossary
related:
  - Autoregressive model
  - ARMA
  - White noise
  - Autocorrelation function
---
A **moving-average** model of order $q$, MA($q$), expresses $Y_t$ as a finite filter of past shocks:

$$
Y_t = \mu + w_t + \phi_1 w_{t-1}+\cdots+\phi_q w_{t-q},
$$

with [$w_t$](/cs-glossary/white-noise/) white noise. Identification: [ACF](/cs-glossary/autocorrelation-function/) cuts off after lag $q$; [PACF](/cs-glossary/partial-autocorrelation/) decays. Combined with AR terms ⇒ [ARMA](/cs-glossary/arma/).
