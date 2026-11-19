---
title: Conditional probability
synonyms:
  - Conditional prob
keywords: conditional probability, P(F|E), Bayes, independence, statistics
related:
  - Independence
  - Bayes' theorem
  - Law of total probability
  - Prior
  - Posterior
---
**Conditional probability** of $F$ given $E$ is

$$
P(F\mid E)=\frac{P(E\cap F)}{P(E)}\quad(P(E)>0).
$$

Equivalently $P(E\cap F)=P(E)\,P(F\mid E)$. Conditioning updates the [sample space](/cs-glossary/sample-space/) to outcomes in $E$. Direction matters: $P(E\mid F)\neq P(F\mid E)$ in general — use [Bayes' theorem](/cs-glossary/bayes-theorem/) to reverse.

[Independence](/cs-glossary/independence/) is the special case $P(F\mid E)=P(F)$.
