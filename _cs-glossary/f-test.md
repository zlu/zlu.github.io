---
title: F-test
synonyms:
  - F statistic
  - Regression F-test
  - Joint significance test
keywords: F-test, F statistic, joint significance, ANOVA, multiple regression, statistics glossary
related:
  - R-squared
  - Model sum of squares
  - Residual sum of squares
  - T-test
  - Hypothesis testing
  - Adjusted R-squared
---
The regression **$F$-test** is a joint test that all slopes are zero: $H_0:\beta_1=\cdots=\beta_k=0$ (intercept free). With Normal iid errors,

$$
F=\frac{S_m^2/k}{S_e^2\big/\bigl[n-(k+1)\bigr]}
=\frac{\mathrm{MSS}/k}{\mathrm{RSS}/(n-k-1)}
\sim F_{k,\,n-k-1}
\quad\text{under }H_0.
$$

Large $F$ when [MSS](/cs-glossary/model-sum-of-squares/) is large relative to [RSS](/cs-glossary/residual-sum-of-squares/), or when $k$ is small. In simple regression ($k=1$), $F=t^2$ for the slope [$t$-test](/cs-glossary/t-test/). Global check of explanatory power; does not validate causality or individual coefficients.
