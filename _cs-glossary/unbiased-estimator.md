---
title: Unbiased estimator
synonyms:
  - Unbiasedness
keywords: unbiased estimator, unbiasedness, BLUE, Gauss-Markov, statistics
related:
  - Bias
  - Mean squared error
  - Gauss–Markov theorem
  - BLUE
  - Sample variance
---
An **unbiased estimator** satisfies $E_\theta[W]=\theta$ for every $\theta$ in the parameter space. Then [MSE](/cs-glossary/mean-squared-error/) equals variance.

Examples: $\bar X$ for a mean; $S^2=(n-1)^{-1}\sum(X_i-\bar X)^2$ for Normal variance. Unbiasedness is a finite-sample property — [consistency](/cs-glossary/consistency/) is the large-sample analogue. Among unbiased linear estimators of a regression slope, [Gauss–Markov](/cs-glossary/gauss-markov-theorem/) picks the minimum-variance one ([BLUE](/cs-glossary/blue/)).
