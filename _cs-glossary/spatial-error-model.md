---
title: Spatial error model
synonyms:
  - SEM
  - Spatial error regression
keywords: spatial error model, SEM, spatially correlated errors, statistics glossary
related:
  - Spatial lag model
  - Spatial weights matrix
  - Spatial autocorrelation
  - Moran's I
  - Ordinary least squares
---
A **spatial error model** (SEM) keeps the mean equation $y=X\beta+u$ but lets disturbances be spatially dependent:

$$
u=\lambda W u + \varepsilon.
$$

Use when omitted spatially structured shocks drive residual correlation rather than substantive spillover in $y$. Lagrange multiplier tests help choose SEM vs [spatial lag](/cs-glossary/spatial-lag-model/) after [Moran's I](/cs-glossary/morans-i/) rejects independence.
