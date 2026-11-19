---
title: Exogeneity
synonyms:
  - Exogenous regressor
  - Mean independence
keywords: exogeneity, exogenous, Cov(X,epsilon)=0, OLS unbiased, statistics
related:
  - Endogeneity
  - Ordinary least squares
  - Unbiased estimator
  - Covariance
---
**Exogeneity** (here: regressor exogenous w.r.t. the error) means $\mathrm{Cov}(X_i,\varepsilon_i)=0$ or stronger $E[\varepsilon\mid X]=0$. Together with zero-mean errors, it makes the [OLS](/cs-glossary/ordinary-least-squares/) slope [unbiased](/cs-glossary/unbiased-estimator/):

$$
\hat\beta=\beta+\frac{\mathrm{Cov}(X,\varepsilon)}{\mathrm{Var}(X)}.
$$

Failure is [endogeneity](/cs-glossary/endogeneity/) — omitted variables, simultaneity, measurement error in $X$.
