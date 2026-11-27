---
title: Total sum of squares
synonyms:
  - TSS
  - St²
  - Total variation
keywords: total sum of squares, TSS, St2, R-squared, ANOVA decomposition, statistics glossary
related:
  - Model sum of squares
  - Residual sum of squares
  - R-squared
  - Sample variance
---
**Total sum of squares** (TSS) measures total sample variation of the response about its mean:

$$
S_t^2=\mathrm{TSS}=\sum_{i=1}^n(y_i-\bar y)^2.
$$

In a model with intercept fitted by [OLS](/cs-glossary/ordinary-least-squares/), the ANOVA identity holds:

$$
\mathrm{TSS}=\mathrm{MSS}+\mathrm{RSS}=S_m^2+S_e^2,
$$

with [model](/cs-glossary/model-sum-of-squares/) and [residual](/cs-glossary/residual-sum-of-squares/) sums of squares. [$R^2$](/cs-glossary/r-squared/) $=S_m^2/S_t^2=1-S_e^2/S_t^2$. TSS is also $(n-1)$ times the usual sample variance of $y$.
