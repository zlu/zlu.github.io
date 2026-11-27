---
title: Autocorrelation function
synonyms:
  - ACF
  - Sample ACF
keywords: autocorrelation function, ACF, lag, time series, ARMA, statistics glossary
related:
  - Autocorrelation
  - Partial autocorrelation
  - Lag
  - ARMA
  - White noise
  - Time series
---
The **autocorrelation function** (ACF) at lag $k$ is the correlation of a series with its $k$-step lag:

$$
\mathrm{ACF}(k)=\frac{\sum_{t=k+1}^{n}(Y_t-\bar Y)(Y_{t-k}-\bar Y)}{\sum_{t=1}^{n}(Y_t-\bar Y)^2}.
$$

$\mathrm{ACF}(0)=1$. Spikes outside confidence bands signal [serial correlation](/cs-glossary/autocorrelation/). With [PACF](/cs-glossary/partial-autocorrelation/), guides [AR](/cs-glossary/autoregressive-model/)/[MA](/cs-glossary/moving-average-model/)/[ARMA](/cs-glossary/arma/) order: ACF tails off & PACF cuts at $p$ ⇒ AR($p$); reverse ⇒ MA($q$).
