---
title: Best unbiased estimator
synonyms:
  - UMVUE
  - Minimum variance unbiased estimator
  - MVUE
keywords: best unbiased estimator, UMVUE, MVUE, efficiency, Cramer-Rao, statistics glossary
related:
  - Unbiased estimator
  - Efficiency (statistics)
  - Mean squared error
  - BLUE
  - Estimator
---
A **best unbiased estimator** $W^*$ of $\theta$ is [unbiased](/cs-glossary/unbiased-estimator/) and has minimum variance among all unbiased estimators: $\mathrm{Var}_\theta[W^*]\le\mathrm{Var}_\theta[W]$ for every unbiased $W$ and all $\theta$.

Also called UMVUE / MVUE. [BLUE](/cs-glossary/blue/) is the same idea restricted to *linear* estimators (Gauss–Markov). Unbiased + min variance $\Rightarrow$ min [MSE](/cs-glossary/mean-squared-error/) inside the unbiased class.
