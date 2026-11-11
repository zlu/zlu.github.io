---
title: Polynomial time
synonyms:
  - Poly time
related:
  - P
  - NP
  - Reduction
  - Decision problem
---
**Polynomial time** means running time bounded by a polynomial in the input length: $O(n^k)$ for some constant $k$.

[P](/cs-glossary/p/) is exactly the [decision problems](/cs-glossary/decision-problem/) with deterministic poly-time algorithms. [NP](/cs-glossary/np/) uses poly-time *verification* of a certificate. Poly-time [reductions](/cs-glossary/reduction/) are the standard way to compare hardness inside these classes.

In practice “poly time” is the theory cut between scalable algorithms and combinatorial explosion. A high-degree polynomial can still be slow; exponential is the usual wall for NP-complete search.
