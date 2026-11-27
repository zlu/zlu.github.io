---
title: Log-log model
synonyms:
  - Log-log regression
  - Double-log model
  - Constant elasticity model
keywords: log-log model, double log, elasticity, linear regression, statistics glossary
related:
  - Elasticity
  - Log-linear model
  - Lin-log model
  - Linear regression
---
A **log-log** regression transforms both sides:

$$
\log Y_i=\alpha+\beta\log X_i+\varepsilon_i.
$$

Still [linear in parameters](/cs-glossary/linear-regression/), so OLS applies. Slope $\beta$ is the [elasticity](/cs-glossary/elasticity/) of $Y$ w.r.t. $X$. Common for multiplicative relationships $Y\approx A X^\beta$.
