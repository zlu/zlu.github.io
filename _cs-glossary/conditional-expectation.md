---
title: Conditional expectation
synonyms:
  - Conditional mean
  - Regression function
keywords: conditional expectation, E[Y|X], regression function, statistics
related:
  - Expectation
  - Linear regression
  - Exogeneity
  - Ordinary least squares
---
**Conditional expectation** $E[Y\mid X=x]$ is the mean of $Y$ given $X=x$. A regression model often posits $E[Y\mid X]=\alpha+\beta X$ (or a richer linear predictor), so fitted values estimate the conditional mean function.

$E[\varepsilon\mid X]=0$ with $\varepsilon=Y-E[Y\mid X]$ encodes correct specification of that mean. Tower property and $E[E[Y\mid X]]=E[Y]$ link conditional and unconditional means.
