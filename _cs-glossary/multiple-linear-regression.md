---
title: Multiple linear regression
synonyms:
  - Multiple regression
  - Multivariable linear regression
keywords: multiple linear regression, multiple regression, partial effect, OLS, statistics glossary
related:
  - Linear regression
  - Ordinary least squares
  - Slope parameter
  - F-test
  - Adjusted R-squared
  - Multicollinearity
---
**Multiple linear regression** extends the simple model to $k$ covariates:

$$
y_i=\beta_0+\beta_1 x_{i1}+\cdots+\beta_k x_{ik}+\varepsilon_i.
$$

Each $\beta_j=\partial E[Y\mid X]/\partial X_j$ is a **partial effect** holding other $X$'s fixed. Fit by [OLS](/cs-glossary/ordinary-least-squares/); joint significance via [$F$](/cs-glossary/f-test/); prefer [adjusted $R^2$](/cs-glossary/adjusted-r-squared/) when comparing $k$. Watch [multicollinearity](/cs-glossary/multicollinearity/).
