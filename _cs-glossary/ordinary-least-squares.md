---
title: Ordinary least squares
synonyms:
  - OLS
  - Least squares
  - Linear least squares
keywords: ordinary least squares, OLS, least squares, linear regression, normal equations, statistics glossary, probability
related:
  - Linear regression
  - Normal equations
  - Residual
  - Fitted value
  - Gauss–Markov theorem
  - BLUE
  - R-squared
  - Residual standard error
---
**Ordinary least squares** (OLS) estimates linear model parameters by minimising residual sum of squares:

$$
\min_{\alpha,\beta}\sum_{i=1}^n(y_i-\alpha-\beta x_i)^2.
$$

The [normal equations](/cs-glossary/normal-equations/) give the closed form $\hat\beta=S_{xy}/S_{xx}$, $\hat\alpha=\bar y-\hat\beta\bar x$. Under [Gauss–Markov](/cs-glossary/gauss-markov-theorem/) assumptions, OLS is [BLUE](/cs-glossary/blue/). With Normal errors, OLS $=$ [MLE](/cs-glossary/maximum-likelihood/) and exact [$t$](/cs-glossary/t-test/) inference applies; report [RSE](/cs-glossary/residual-standard-error/), [$R^2$](/cs-glossary/r-squared/), and [$F$](/cs-glossary/f-test/) alongside coefficient SEs.
