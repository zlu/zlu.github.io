---
title: Linear regression
synonyms:
  - Simple linear regression
  - Linear model
keywords: linear regression, simple OLS, conditional mean, regression function, statistics glossary, probability
related:
  - Regression function
  - Ordinary least squares
  - Dependent variable
  - Independent variable
  - Intercept
  - Slope parameter
  - Error term
  - Conditional expectation
  - Residual
  - Exogeneity
  - R-squared
---
**Linear regression** models a [response](/cs-glossary/dependent-variable/) $Y$ as linear *in parameters* in [covariates](/cs-glossary/independent-variable/) $X$, plus an [error](/cs-glossary/error-term/):

$$
y_i=\alpha+\beta x_i+\varepsilon_i
\quad\Leftrightarrow\quad
E[Y_i\mid X_i]=\alpha+\beta X_i
\text{ when }E[\varepsilon_i\mid X_i]=0.
$$

The [regression function](/cs-glossary/regression-function/) is $\mu(X,\beta)=\beta^\top X$. [Intercept](/cs-glossary/intercept/) $\alpha$ is mean $Y$ at $X=0$; [slope](/cs-glossary/slope-parameter/) $\beta=\partial E[Y\mid X]/\partial X$. “Linear” allows transformed regressors ($x^2$, $\log x$) but not nonlinearity in $\beta$ (e.g. $\beta^2 x$). Fit with [OLS](/cs-glossary/ordinary-least-squares/); judge fit with [RSE](/cs-glossary/residual-standard-error/), [$R^2$](/cs-glossary/r-squared/), and [$F$](/cs-glossary/f-test/).
