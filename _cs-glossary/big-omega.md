---
title: Big-Omega
synonyms:
  - Big Omega
  - Ω notation
related:
  - Big-O
  - Big-Theta
  - Lower bound
  - Time complexity
---
**Big-Omega** is an asymptotic lower bound: $f(n) = \Omega(g(n))$ means $f$ grows at least as fast as a constant times $g$ for large $n$.

Use it for “any algorithm needs at least …” results, such as $\Omega(n \log n)$ comparisons for [comparison sorting](/cs-glossary/comparison-sort/) in the decision-tree model ([lower bound](/cs-glossary/lower-bound/)). An algorithm’s own running-time proof more often states $O$ or $\Theta$.

If $f = O(g)$ and $f = \Omega(g)$, then $f = \Theta(g)$.
