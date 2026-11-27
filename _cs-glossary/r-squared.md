---
title: R-squared
synonyms:
  - R²
  - Coefficient of determination
  - R2
keywords: R-squared, R², coefficient of determination, goodness of fit, MSS, TSS, statistics glossary
related:
  - Residual sum of squares
  - Total sum of squares
  - Model sum of squares
  - Adjusted R-squared
  - Residual standard error
  - F-test
  - Ordinary least squares
---
**$R^2$** (coefficient of determination) is the share of sample variation in $Y$ captured by the fitted model:

$$
R^2=\frac{S_m^2}{S_t^2}=\frac{\mathrm{MSS}}{\mathrm{TSS}}=1-\frac{S_e^2}{S_t^2}=1-\frac{\mathrm{RSS}}{\mathrm{TSS}},
$$

with [MSS](/cs-glossary/model-sum-of-squares/) $=\sum(\hat y_i-\bar y)^2$, [TSS](/cs-glossary/total-sum-of-squares/) $=\sum(y_i-\bar y)^2$, and [RSS](/cs-glossary/residual-sum-of-squares/) $=\sum e_i^2$. In models with an intercept fitted by [OLS](/cs-glossary/ordinary-least-squares/), $R^2\in[0,1]$ and $\mathrm{TSS}=\mathrm{MSS}+\mathrm{RSS}$.

Relative fit (unit-free). Contrast [RSE](/cs-glossary/residual-standard-error/) $s$, which is absolute and in units of $Y$. Prefer [adjusted $R^2$](/cs-glossary/adjusted-r-squared/) when comparing models with different $k$. High $R^2\neq$ causal truth; low $R^2\neq$ useless slope — still check the [$t$](/cs-glossary/t-test/) on $\beta$.
