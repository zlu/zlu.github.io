---
title: Regression function
synonyms:
  - Conditional mean function
  - μ(X, β)
keywords: regression function, conditional expectation, mu, linear model, statistics glossary
related:
  - Conditional expectation
  - Linear regression
  - Dependent variable
  - Error term
  - Fitted value
---
The **regression function** is the parametric model for the conditional mean of the [response](/cs-glossary/dependent-variable/):

$$
E[Y\mid X]=\mu(X,\beta).
$$

It is the systematic part of $Y=\mu(X,\beta)+\varepsilon$. In this course’s linear model, $\mu$ is linear *in parameters*: $\mu(X,\beta)=\beta^\top X$ (possibly after transforming or expanding covariates — polynomials, logs, interactions). The [fitted value](/cs-glossary/fitted-value/) $\hat y=\mu(x,\hat\beta)$ plugs in [OLS](/cs-glossary/ordinary-least-squares/) (or MLE) estimates of $\beta$.
