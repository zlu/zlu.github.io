---
title: Degrees of freedom
synonyms:
  - df
  - Residual degrees of freedom
keywords: degrees of freedom, df, n-1, residual df, statistics
related:
  - Sample variance
  - T-test
  - Ordinary least squares
  - Residual
---
**Degrees of freedom** (df) count independent pieces of information after constraints. In $S^2$, divisor $n-1$ reflects one constraint $\sum(X_i-\bar X)=0$ from using $\bar X$.

In simple [OLS](/cs-glossary/ordinary-least-squares/) with intercept and slope, residual df is $n-2$, so $s^2=\mathrm{RSS}/(n-2)$ and $t$-critical values use $t_{n-2}$. More parameters fitted ⇒ fewer residual df.
