---
title: Likelihood
synonyms:
  - Likelihood function
keywords: likelihood, likelihood function, MLE, Bayes, statistics glossary
related:
  - Maximum likelihood
  - Prior
  - Posterior
  - Bayes' theorem
  - Estimator
---
**Likelihood** $L(\theta;x)$ treats the observed data $x$ as fixed and the parameter $\theta$ as the variable: for iid data, $L(\theta;x)=\prod_i f(x_i;\theta)$. It is *not* a probability distribution over $\theta$ unless normalised in a Bayesian model.

[Maximum likelihood](/cs-glossary/maximum-likelihood/) maximises $L$ (or $\ell=\log L$). In [Bayes' theorem](/cs-glossary/bayes-theorem/), likelihood is $P(\text{data}\mid\text{hypothesis})$ — e.g. sensitivity $P(T\mid D)$.
