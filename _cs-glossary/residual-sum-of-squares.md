---
title: Residual sum of squares
synonyms:
  - RSS
  - Sum of squared residuals
  - SSE
  - Se²
keywords: residual sum of squares, RSS, SSE, Se2, R-squared, RSE, statistics glossary
related:
  - Residual
  - Residual standard error
  - R-squared
  - Total sum of squares
  - Model sum of squares
  - Ordinary least squares
  - Normal equations
---
**Residual sum of squares** (RSS), denoted $S_e^2$ in the lecture notes, is

$$
S_e^2=\mathrm{RSS}=\sum_{i=1}^n e_i^2=\sum_{i=1}^n(y_i-\hat y_i)^2=\sum_{i=1}^n(y_i-\alpha-\beta x_i)^2
$$

evaluated at the chosen $(\alpha,\beta)$. [Least squares](/cs-glossary/ordinary-least-squares/) minimises RSS via the [normal equations](/cs-glossary/normal-equations/). Unbiased error variance in simple regression: $s^2=\mathrm{RSS}/(n-2)$; [RSE](/cs-glossary/residual-standard-error/) is $s=\sqrt{s^2}$.

[$R^2$](/cs-glossary/r-squared/) $=1-\mathrm{RSS}/\mathrm{TSS}$. Lower RSS means better in-sample fit — not automatically a better causal model.
