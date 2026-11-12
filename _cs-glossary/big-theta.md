---
title: Big-Theta
synonyms:
  - Big Theta
  - Θ notation
related:
  - Big-O
  - Big-Omega
  - Time complexity
---
**Big-Theta** is a tight asymptotic bound: $f(n) = \Theta(g(n))$ means $f = O(g)$ and $f = \Omega(g)$. Growth matches $g$ up to constant factors.

Prefer $\Theta$ in exam writeups when both directions are clear: [merge sort](/cs-glossary/merge-sort/) is $\Theta(n \log n)$, not merely $O(n \log n)$. Reserve plain [Big-O](/cs-glossary/big-o/) when you only proved an upper bound (or when stating a loose ceiling on purpose).
