---
title: Residual
synonyms:
  - OLS residual
  - Fitting residual
  - ei
keywords: residual, OLS residual, RSS, RSE, regression diagnostics, statistics glossary
related:
  - Fitted value
  - Error term
  - Ordinary least squares
  - Residual sum of squares
  - Residual standard error
  - Homoscedasticity
  - Durbin–Watson
---
A **residual** in [OLS](/cs-glossary/ordinary-least-squares/) is the gap between the observed response and the [fitted value](/cs-glossary/fitted-value/):

$$
e_i=y_i-\hat y_i=y_i-\hat\alpha-\hat\beta x_i.
$$

Residuals estimate the unobservable [errors](/cs-glossary/error-term/) $\varepsilon_i$. RSS $=\sum e_i^2$; [RSE](/cs-glossary/residual-standard-error/) $s=\sqrt{\mathrm{RSS}/(n-2)}$. Residual diagnostics (plots, [Durbin–Watson](/cs-glossary/durbin-watson/)) check [homoscedasticity](/cs-glossary/homoscedasticity/), dependence, and nonlinearity. Residual df is typically $n-p$.
