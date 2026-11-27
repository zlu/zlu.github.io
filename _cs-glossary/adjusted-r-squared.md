---
title: Adjusted R-squared
synonyms:
  - Adjusted R²
  - Adj. R²
  - R-bar-squared
keywords: adjusted R-squared, adjusted R², goodness of fit, multiple regression, statistics glossary
related:
  - R-squared
  - Residual sum of squares
  - Total sum of squares
  - F-test
  - Ordinary least squares
---
**Adjusted $R^2$** is a degrees-of-freedom–penalised version of [$R^2$](/cs-glossary/r-squared/). For a linear model with $k$ covariates (plus intercept),

$$
\bar R^2=1-\frac{\mathrm{RSS}/(n-k-1)}{\mathrm{TSS}/(n-1)}.
$$

Raw $R^2$ never falls when you add a regressor (RSS weakly decreases). The residual variance estimator $\mathrm{RSS}/(n-k-1)$ can rise or fall, so adjusted $R^2$ rewards covariates that buy real explanatory power and can drop when a variable is superfluous. Still in-sample and not a causal warrant.
