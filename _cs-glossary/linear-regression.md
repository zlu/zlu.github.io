---
title: Linear regression
synonyms:
  - Simple linear regression
  - Linear model
keywords: linear regression, simple OLS, conditional mean, statistics glossary, probability
related:
  - Ordinary least squares
  - Conditional expectation
  - Residual
  - Exogeneity
  - R-squared
---
**Linear regression** models a response $Y$ as linear *in parameters* in covariates $X$, plus error: $y_i=\alpha+\beta x_i+\varepsilon_i$ in the simple case. Equivalently $Y=E[Y\mid X]+\varepsilon$ with $E[\varepsilon\mid X]=0$.

“Linear” allows transformed regressors ($x^2$, $\log x$) but not nonlinearity in $\beta$ (e.g. $\beta^2 x$). Fit with [OLS](/cs-glossary/ordinary-least-squares/); interpret $\beta$ as $\partial E[Y\mid X]/\partial X$ when the model is correct.
