---
title: Ordinary least squares
synonyms:
  - OLS
  - Least squares
  - Linear least squares
keywords: ordinary least squares, OLS, least squares, linear regression, statistics glossary, probability
related:
  - Linear regression
  - Residual
  - Gauss–Markov theorem
  - BLUE
  - R-squared
---
**Ordinary least squares** (OLS) estimates linear model parameters by minimising residual sum of squares:

$$
\min_{\alpha,\beta}\sum_{i=1}^n(y_i-\alpha-\beta x_i)^2.
$$

Closed form: $\hat\beta=S_{xy}/S_{xx}$, $\hat\alpha=\bar y-\hat\beta\bar x$. Under [Gauss–Markov](/cs-glossary/gauss-markov-theorem/) assumptions, OLS is [BLUE](/cs-glossary/blue/). With Normal errors, OLS $=$ [MLE](/cs-glossary/maximum-likelihood/) and exact [$t$](/cs-glossary/t-test/) inference applies.
