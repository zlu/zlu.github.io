---
title: Variance inflation factor
synonyms:
  - VIF
keywords: variance inflation factor, VIF, multicollinearity, diagnostics, statistics glossary
related:
  - Multicollinearity
  - Multiple linear regression
  - Standard error
---
The **variance inflation factor** for regressor $j$ is

$$
\mathrm{VIF}_j=\frac{1}{1-R_j^2},
$$

where $R_j^2$ is from regressing $X_j$ on the other covariates. $\mathrm{VIF}=1$ means no linear dependence; large VIF (rules of thumb: $>5$ or $>10$) flags [multicollinearity](/cs-glossary/multicollinearity/) inflating $\mathrm{Var}(\hat\beta_j)$.
