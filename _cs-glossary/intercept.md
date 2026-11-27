---
title: Intercept
synonyms:
  - Intercept parameter
  - α
  - β₀
keywords: intercept, alpha, beta0, linear regression, OLS, statistics glossary
related:
  - Slope parameter
  - Linear regression
  - Ordinary least squares
  - Fitted value
---
The **intercept** $\alpha$ (or $\beta_0$) is the constant term in a [linear regression](/cs-glossary/linear-regression/). It is the mean of $Y$ when all covariates equal zero:

$$
\alpha=E[Y\mid X_1=\cdots=X_k=0].
$$

[OLS](/cs-glossary/ordinary-least-squares/) gives $\hat\alpha=\bar y-\hat\beta\bar x$ in the simple model, so the fitted line always passes through $(\bar x,\bar y)$. Without an intercept the line is forced through the origin. Substantive interpretation of $\hat\alpha$ only matters when $x=0$ is in range and meaningful; often the slope is the quantity of interest.
