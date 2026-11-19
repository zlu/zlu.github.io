---
title: Maximum likelihood
synonyms:
  - MLE
  - Maximum likelihood estimation
  - Maximum likelihood estimator
keywords: maximum likelihood, MLE, log-likelihood, estimator, statistics glossary, probability
related:
  - Likelihood
  - Method of moments
  - Estimator
  - Consistency
  - Asymptotic normality
---
**Maximum likelihood estimation** (MLE) chooses $\hat\theta$ to maximise the [likelihood](/cs-glossary/likelihood/) $L(\theta;x)=\prod_i f(x_i;\theta)$ for fixed data $x$. Prefer the log-likelihood $\ell=\log L$; solve $\partial\ell/\partial\theta_j=0$ and check a maximum.

Examples: Bernoulli $\hat p=\bar X$; Normal $\hat\mu=\bar x$, $\hat\sigma^2_{\mathrm{MLE}}=n^{-1}\sum(x_i-\bar x)^2$ (biased for $\sigma^2$). Under regularity, MLEs are typically [consistent](/cs-glossary/consistency/) and [asymptotically normal](/cs-glossary/asymptotic-normality/). For Normal linear models with Normal errors, MLE coincides with [OLS](/cs-glossary/ordinary-least-squares/).
