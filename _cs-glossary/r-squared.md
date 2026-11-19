---
title: R-squared
synonyms:
  - R²
  - Coefficient of determination
  - R2
keywords: R-squared, R², coefficient of determination, goodness of fit, statistics
related:
  - Residual sum of squares
  - Ordinary least squares
  - Correlation
  - T-test
---
**$R^2$** (coefficient of determination) is $1-\mathrm{RSS}/\mathrm{TSS}$, the fraction of sample variance in $Y$ “explained” by the fitted linear predictor. $R^2\in[0,1]$ in models with an intercept fitted by [OLS](/cs-glossary/ordinary-least-squares/).

High $R^2$ $\neq$ causal truth; low $R^2$ $\neq$ useless slope — still check the [$t$](/cs-glossary/t-test/) on $\beta$. In-sample fit can be high while [endogeneity](/cs-glossary/endogeneity/) wrecks interpretation.
