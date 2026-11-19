---
title: Endogeneity
synonyms:
  - Endogenous regressor
keywords: endogeneity, endogenous, OLS bias, omitted variable, statistics
related:
  - Exogeneity
  - Bias
  - Consistency
  - Ordinary least squares
---
**Endogeneity** means a regressor is correlated with the error: $\mathrm{Cov}(X,\varepsilon)\neq 0$. Then [OLS](/cs-glossary/ordinary-least-squares/) $\hat\beta$ is [biased](/cs-glossary/bias/) and inconsistent for the structural slope.

Classic causes: omitted confounders, reverse causality / simultaneity, measurement error in $X$. Fixes lie outside plain OLS (instruments, experiments, richer models) — not more in-sample $R^2$.
