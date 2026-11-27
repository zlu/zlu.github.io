---
title: Moment generating function
synonyms:
  - MGF
  - Moment-generating function
keywords: moment generating function, MGF, moments, Laplace transform, statistics glossary, probability
related:
  - Moment
  - Expectation
  - Likelihood
  - Normal distribution
---
The **moment generating function** (MGF) of $X$ is

$$
M_X(t)=E[e^{tX}],
$$

when the expectation exists in a neighbourhood of $0$. Differentiating at the origin recovers raw moments: $E[X^n]=M_X^{(n)}(0)$.

MGFs uniquely determine distributions (when they exist) and turn sums of independent RVs into products of MGFs — handy for [Normal](/cs-glossary/normal-distribution/) / exponential-family calculations.
