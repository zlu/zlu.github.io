---
title: Independent variable
synonyms:
  - Explanatory variable
  - Covariate
  - Regressor
  - Predictor
  - X
keywords: independent variable, covariate, regressor, explanatory variable, regression, statistics glossary
related:
  - Dependent variable
  - Exogeneity
  - Linear regression
  - Slope parameter
---
An **independent variable** (explanatory variable / covariate / regressor), denoted $X$, is an observable used to explain variation in the [dependent variable](/cs-glossary/dependent-variable/) $Y$. In the linear model $y_i=\beta_0+\sum_j\beta_j X_{ij}+\varepsilon_i$, each $X_j$ enters the [regression function](/cs-glossary/regression-function/); its [slope](/cs-glossary/slope-parameter/) $\beta_j=\partial E[Y\mid X]/\partial X_j$ is a partial effect holding other covariates fixed.

“Independent” here means *explanatory*, not statistical independence of $X$ from $\varepsilon$ — that stronger condition is [exogeneity](/cs-glossary/exogeneity/) / $E[\varepsilon\mid X]=0$.
