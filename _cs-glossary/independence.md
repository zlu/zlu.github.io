---
title: Independence
synonyms:
  - Statistical independence
  - Independent events
keywords: independence, independent events, uncorrelated, iid, probability, statistics
related:
  - Conditional probability
  - Covariance
  - Correlation
  - IID
  - Random sample
---
Two events $E,F$ are **independent** if $P(F\mid E)=P(F)$ (when $P(E)>0$), equivalently $P(E\cap F)=P(E)P(F)$. Knowing $E$ occurred does not change the probability of $F$.

For [random variables](/cs-glossary/random-variable/), independence means the joint factors into the product of [marginals](/cs-glossary/marginal-distribution/). Independence implies [covariance](/cs-glossary/covariance/) zero; the converse is false (uncorrelated $\neq$ independent).

An [iid](/cs-glossary/iid/) [random sample](/cs-glossary/random-sample/) is mutually independent with a common distribution.
