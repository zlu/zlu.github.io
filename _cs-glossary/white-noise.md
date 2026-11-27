---
title: White noise
synonyms:
  - WN
  - Innovation
  - Idiosyncratic shock
keywords: white noise, innovation, iid errors, ARMA residual, statistics glossary
related:
  - IID
  - Autocorrelation
  - ARMA
  - Residual
  - Stationarity
---
**White noise** $\{w_t\}$ is an uncorrelated (often iid) zero-mean, constant-variance sequence: $E[w_t]=0$, $\mathrm{Var}(w_t)=\sigma^2$, $\mathrm{Cov}(w_t,w_{s})=0$ for $t\neq s$. Often assumed Normal.

[ARMA](/cs-glossary/arma/) residuals should behave like white noise — check [ACF](/cs-glossary/autocorrelation-function/) of residuals. [IID](/cs-glossary/iid/) $\Rightarrow$ white noise; white noise $\nRightarrow$ iid without extra assumptions.
