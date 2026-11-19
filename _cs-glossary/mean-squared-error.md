---
title: Mean squared error
synonyms:
  - MSE
  - Quadratic risk
keywords: mean squared error, MSE, bias variance tradeoff, estimator, statistics
related:
  - Bias
  - Variance
  - Estimator
  - Consistency
---
**Mean squared error** of an [estimator](/cs-glossary/estimator/) $W$ for $\theta$ is

$$
\mathrm{MSE}_\theta(W)=E_\theta[(W-\theta)^2]=\mathrm{Var}_\theta(W)+\mathrm{Bias}_\theta(W)^2.
$$

It scores overall accuracy: variance and squared bias. Unbiased ⇒ MSE $=$ Var. Bias–variance trade-offs are common (e.g. Normal variance MLE vs $S^2$ at small $n$).
