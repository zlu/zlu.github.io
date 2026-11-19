---
title: Bias
synonyms:
  - Estimator bias
  - Statistical bias
keywords: bias, unbiased, Bias(W), MSE, estimator, statistics
related:
  - Unbiased estimator
  - Mean squared error
  - Variance
  - Consistency
  - Exogeneity
---
**Bias** of an [estimator](/cs-glossary/estimator/) $W$ for $\theta$ is $\mathrm{Bias}_\theta(W)=E_\theta[W]-\theta$. **Unbiased** means $E_\theta[W]=\theta$ for all $\theta$ in the model.

[MSE](/cs-glossary/mean-squared-error/) $=\mathrm{Var}+\mathrm{Bias}^2$, so a little bias can buy lower variance. Normal $\hat\sigma^2_{\mathrm{MLE}}$ has bias $-\sigma^2/n$; bias-corrected $S^2$ is unbiased. For [OLS](/cs-glossary/ordinary-least-squares/) slopes, zero-mean errors and [exogeneity](/cs-glossary/exogeneity/) ($\mathrm{Cov}(X,\varepsilon)=0$) deliver unbiasedness.
