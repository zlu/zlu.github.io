---
title: Spatial lag model
synonyms:
  - SLM
  - Spatial autoregressive model
  - SAM
  - SAR
keywords: spatial lag model, SLM, SAR, spatially lagged dependent variable, statistics glossary
related:
  - Spatial error model
  - Spatial weights matrix
  - Spatial autocorrelation
  - Moran's I
  - Ordinary least squares
---
A **spatial lag model** (SLM / SAR) puts a spatially lagged *outcome* on the right-hand side:

$$
y=\rho W y + Xb + \varepsilon.
$$

$\rho$ is the spatial autoregressive parameter; $Wy$ is the weighted average of neighbours’ $y$. Use when outcomes spill over across space. Effects decompose into direct and indirect (spillover) impacts via $(I-\rho W)^{-1}$.
