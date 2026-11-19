---
title: Law of total probability
synonyms:
  - Total probability
  - Total probability rule
keywords: law of total probability, total probability, Bayes denominator, statistics
related:
  - Conditional probability
  - Bayes' theorem
  - Prior
  - Posterior
---
The **law of total probability** expands an unconditional probability by conditioning on a partition. For event $E$ and its complement:

$$
P(F)=P(F\mid E)P(E)+P(F\mid E')P(E').
$$

More generally, if $E_1,\ldots,E_k$ partition $\Omega$, then $P(F)=\sum_i P(F\mid E_i)P(E_i)$. This supplies the denominator in [Bayes' theorem](/cs-glossary/bayes-theorem/) (e.g. diagnostic tests: disease vs no-disease).
