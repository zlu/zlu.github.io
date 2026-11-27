---
title: Model sum of squares
synonyms:
  - MSS
  - ESS
  - Explained sum of squares
  - Sm²
  - Regression sum of squares
keywords: model sum of squares, MSS, ESS, explained sum of squares, R-squared, F-test, statistics glossary
related:
  - Total sum of squares
  - Residual sum of squares
  - R-squared
  - F-test
  - Fitted value
---
**Model sum of squares** (MSS; also ESS / regression SS) is the variation in $Y$ “explained” by the fitted line — how far fitted values sit from $\bar y$:

$$
S_m^2=\mathrm{MSS}=\sum_{i=1}^n(\hat y_i-\bar y)^2.
$$

Together with [RSS](/cs-glossary/residual-sum-of-squares/) it partitions [TSS](/cs-glossary/total-sum-of-squares/): $\mathrm{TSS}=\mathrm{MSS}+\mathrm{RSS}$. Then [$R^2$](/cs-glossary/r-squared/) $=\mathrm{MSS}/\mathrm{TSS}$. The regression [$F$-test](/cs-glossary/f-test/) uses $\mathrm{MSS}/k$ in the numerator against residual mean square.
