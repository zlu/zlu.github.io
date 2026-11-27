---
title: Error term
synonyms:
  - Disturbance
  - Disturbance term
  - ε
keywords: error term, disturbance, epsilon, regression, exogeneity, statistics glossary
related:
  - Residual
  - Exogeneity
  - Homoscedasticity
  - Regression function
  - Dependent variable
---
The **error term** (disturbance) $\varepsilon$ makes the link between $Y$ and $X$ stochastic rather than deterministic. In conditional-expectation form,

$$
Y=E[Y\mid X]+\varepsilon,\qquad E[\varepsilon\mid X]=0.
$$

Typical sources: natural randomness, omitted variables, measurement error, wrong functional form. The unobservable $\varepsilon_i$ is estimated by the [residual](/cs-glossary/residual/) $e_i=y_i-\hat y_i$. Gauss–Markov needs $E[\varepsilon_i]=0$, $\mathrm{Cov}(X_i,\varepsilon_i)=0$, [homoscedastic](/cs-glossary/homoscedasticity/) $\mathrm{Var}(\varepsilon_i)=\sigma^2$, and $\mathrm{Cov}(\varepsilon_i,\varepsilon_j)=0$ for $i\neq j$.
