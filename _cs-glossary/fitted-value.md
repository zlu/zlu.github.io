---
title: Fitted value
synonyms:
  - Predicted value
  - Expected value (fitted)
  - ŷ
keywords: fitted value, predicted value, y-hat, OLS, regression, statistics glossary
related:
  - Residual
  - Linear regression
  - Ordinary least squares
  - Model sum of squares
---
A **fitted value** (or in-sample predicted value) is the regression function evaluated at the estimated parameters. For simple [linear regression](/cs-glossary/linear-regression/) $\mu(x)=\alpha+\beta x$,

$$
\hat y_i=\hat\alpha+\hat\beta x_i.
$$

The [residual](/cs-glossary/residual/) is $e_i=y_i-\hat y_i$. [OLS](/cs-glossary/ordinary-least-squares/) with an intercept forces the fitted line through $(\bar x,\bar y)$, so $\sum e_i=0$ and $\overline{\hat y}=\bar y$. [MSS](/cs-glossary/model-sum-of-squares/) is built from $(\hat y_i-\bar y)^2$.
