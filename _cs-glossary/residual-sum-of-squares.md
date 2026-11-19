---
title: Residual sum of squares
synonyms:
  - RSS
  - Sum of squared residuals
  - SSE
keywords: residual sum of squares, RSS, SSE, R-squared, statistics
related:
  - Residual
  - R-squared
  - Ordinary least squares
  - Standard error
---
**Residual sum of squares** (RSS) is $\sum_{i=1}^n e_i^2$, the sum of squared [OLS](/cs-glossary/ordinary-least-squares/) residuals. Least squares minimises RSS. The unbiased error variance estimator in simple regression is $s^2=\mathrm{RSS}/(n-2)$.

[$R^2$](/cs-glossary/r-squared/) $=1-\mathrm{RSS}/\mathrm{TSS}$ compares RSS to total sum of squares. Lower RSS means better in-sample fit — not automatically a better causal model.
