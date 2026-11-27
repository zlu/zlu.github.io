---
title: Marginal likelihood
synonyms:
  - Evidence
  - Prior predictive
keywords: marginal likelihood, evidence, Bayes, prior predictive, statistics glossary
related:
  - Likelihood
  - Prior
  - Posterior
  - Bayes' theorem
---
In Bayesian inference the **marginal likelihood** (evidence) of the data is

$$
f_X(x)=\int \pi(\theta)\,f_{X\mid\theta}(x;\theta)\,d\theta,
$$

the normalising constant in [Bayes](/cs-glossary/bayes-theorem/)' formula for the [posterior](/cs-glossary/posterior/) $\pi(\theta\mid x)\propto \pi(\theta)f(x\mid\theta)$. It does not depend on $\theta$ and is used in Bayes factors for model comparison.
