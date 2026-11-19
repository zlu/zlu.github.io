---
title: Normal distribution
synonyms:
  - Gaussian distribution
  - Gaussian
  - Bell curve
keywords: normal distribution, Gaussian, N(mu, sigma^2), CLT, statistics
related:
  - Central limit theorem
  - Expectation
  - Variance
  - Probability density function
  - Z-score
---
The **Normal** (Gaussian) distribution $N(\mu,\sigma^2)$ has pdf

$$
f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

Mean $\mu$, variance $\sigma^2$. Match the prefactor and exponent to read parameters — do not integrate. The [CLT](/cs-glossary/central-limit-theorem/) makes averages approximately Normal; OLS inference under Normal errors yields exact [$t$](/cs-glossary/t-test/) procedures.
