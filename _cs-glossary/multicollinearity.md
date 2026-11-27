---
title: Multicollinearity
synonyms:
  - Collinearity
  - Perfect collinearity
keywords: multicollinearity, collinearity, VIF, correlated regressors, statistics glossary
related:
  - Multiple linear regression
  - Variance inflation factor
  - Standard error
  - T-test
  - Independent variable
---
**Multicollinearity** is high linear dependence among regressors. Coefficients remain unbiased under Gauss–Markov, but [SEs](/cs-glossary/standard-error/) inflate and individual [$t$](/cs-glossary/t-test/)/$p$-values become unstable — a covariate can look “insignificant” while the group jointly matters ([$F$](/cs-glossary/f-test/)).

Perfect collinearity (exact linear relation) makes OLS undefined. Diagnose with correlations / [VIF](/cs-glossary/variance-inflation-factor/); remedies: drop/combine variables, more data, regularisation.
