---
title: Probability axioms
synonyms:
  - Kolmogorov axioms
  - Axioms of probability
keywords: probability axioms, Kolmogorov, probability rules, statistics glossary
related:
  - Event
  - Sample space
  - Conditional probability
  - Bayes' theorem
---
**Probability axioms** (Kolmogorov) say a probability $P$ on events must satisfy:

1. $0\le P(E)\le 1$ for every event $E$
2. $P(\Omega)=1$
3. Countable additivity for pairwise disjoint events

Useful consequences: $P(E')=1-P(E)$, $P(\emptyset)=0$, and $P(E\cup F)=P(E)+P(F)-P(E\cap F)$. These rules underwrite [conditional probability](/cs-glossary/conditional-probability/), the [law of total probability](/cs-glossary/law-of-total-probability/), and [Bayes' theorem](/cs-glossary/bayes-theorem/).
