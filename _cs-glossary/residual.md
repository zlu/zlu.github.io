---
title: Residual
synonyms:
  - OLS residual
  - Fitting residual
keywords: residual, OLS residual, RSS, regression diagnostics, statistics
related:
  - Ordinary least squares
  - Residual sum of squares
  - Homoscedasticity
  - Durbin–Watson
---
A **residual** in [OLS](/cs-glossary/ordinary-least-squares/) is $e_i=y_i-\hat\alpha-\hat\beta x_i$ — the gap between observed $y_i$ and the fitted value. Residuals estimate the unobservable errors $\varepsilon_i$.

RSS $=\sum e_i^2$ is the least-squares criterion value. Residual diagnostics (plots, [Durbin–Watson](/cs-glossary/durbin-watson/)) check [homoscedasticity](/cs-glossary/homoscedasticity/), dependence, and nonlinearity. Residual df is typically $n-p$.
