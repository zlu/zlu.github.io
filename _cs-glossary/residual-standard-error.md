---
title: Residual standard error
synonyms:
  - RSE
  - Residual SE
  - s
keywords: residual standard error, RSE, s, goodness of fit, OLS, statistics glossary
related:
  - Residual sum of squares
  - Standard error
  - R-squared
  - Ordinary least squares
  - Degrees of freedom
---
**Residual standard error** (RSE) is the sample standard deviation of the [OLS](/cs-glossary/ordinary-least-squares/) residuals — an absolute measure of average miss in units of $Y$:

$$
s=\sqrt{s^2}=\sqrt{\frac{S_e^2}{n-2}}=\sqrt{\frac{\mathrm{RSS}}{n-2}},
$$

where $S_e^2=\sum_{i=1}^n e_i^2=\mathrm{RSS}$ and $n-2$ are the residual [degrees of freedom](/cs-glossary/degrees-of-freedom/) in simple regression (general: $n-k-1$).

RSE estimates $\sigma=\mathrm{sd}(\varepsilon)$ around the true regression function. Small $s$ means $\hat y_i\approx y_i$; large $s$ means a poor absolute fit. Unlike [$R^2$](/cs-glossary/r-squared/), interpretation needs the scale of $Y$. The same $s$ enters [standard errors](/cs-glossary/standard-error/) of $\hat\alpha$ and $\hat\beta$.
