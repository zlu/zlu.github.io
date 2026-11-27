---
title: Continuous random variable
synonyms:
  - Continuous RV
keywords: continuous random variable, pdf, density, statistics glossary, probability
related:
  - Random variable
  - Probability density function
  - Discrete random variable
  - Cumulative distribution function
---
A **continuous random variable** $X$ takes values in intervals (uncountable support). Probabilities come from a [pdf](/cs-glossary/probability-density-function/) $f_X$ via

$$
P(a<X\le b)=\int_a^b f_X(t)\,dt,
$$

with $P(X=x)=0$ for each single point. The [CDF](/cs-glossary/cumulative-distribution-function/) is $F_X(x)=\int_{-\infty}^x f_X$. Classic example: [Normal](/cs-glossary/normal-distribution/).
