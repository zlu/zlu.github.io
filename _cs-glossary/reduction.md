---
title: Reduction
synonyms:
  - Many-one reduction
  - Polynomial-time reduction
  - Karp reduction
related:
  - NP
  - NP-Complete
  - NP-Hard
  - Decision problem
  - Polynomial time
---
A **reduction** from problem $A$ to problem $B$ is a way to solve $A$ by translating instances into instances of $B$. If the translation is cheap and $B$ is easy, $A$ is easy too.

For NP theory the usual tool is a [polynomial-time](/cs-glossary/polynomial-time/) many-one (Karp) reduction: a poly-time function $f$ such that $x$ is a yes-instance of $A$ iff $f(x)$ is a yes-instance of $B$. Write $A \leq_p B$.

[NP-hard](/cs-glossary/np-hard/) means every problem in [NP](/cs-glossary/np/) reduces to you. [NP-complete](/cs-glossary/np-complete/) means you are in NP and NP-hard. Reductions are how textbooks prove “this puzzle is as hard as SAT.”
