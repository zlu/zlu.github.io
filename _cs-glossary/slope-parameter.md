---
title: Slope parameter
synonyms:
  - Slope
  - Regression coefficient
  - β
  - Partial effect
keywords: slope parameter, beta, regression coefficient, partial effect, OLS, statistics glossary
related:
  - Intercept
  - Independent variable
  - Ordinary least squares
  - T-test
  - Exogeneity
---
A **slope parameter** $\beta_j$ measures how the conditional mean of $Y$ changes with covariate $X_j$. In the linear model,

$$
\beta_j=\frac{\partial E[Y\mid X_1,\ldots,X_k]}{\partial X_j}.
$$

Simple OLS: $\hat\beta=S_{xy}/S_{xx}=\mathrm{Cov}(x,y)/\mathrm{Var}(x)$. Read $\hat\beta$ as a change in average $Y$ per unit $X$ — causal only under [exogeneity](/cs-glossary/exogeneity/) / a credible design. Significance of a single slope uses a [$t$-test](/cs-glossary/t-test/); joint “all slopes zero?” uses an [$F$-test](/cs-glossary/f-test/).
