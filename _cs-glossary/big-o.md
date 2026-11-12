---
title: Big-O
synonyms:
  - Big O notation
  - O notation
  - Asymptotic notation
related:
  - Time complexity
  - Space complexity
  - Big-Omega
  - Big-Theta
  - Lower bound
---
**Big-O** notation writes an asymptotic upper bound: $f(n) = O(g(n))$ means for large $n$, $f$ grows at most as fast as a constant times $g$ (formally $\exists c, n_0$ such that $f(n) \le c\,g(n)$ for $n \ge n_0$).

Coders use it for [time](/cs-glossary/time-complexity/) and [space](/cs-glossary/space-complexity/) of algorithms. $O$ alone is an upper bound; [$\Theta$](/cs-glossary/big-theta/) is tight; [$\Omega$](/cs-glossary/big-omega/) is a lower bound. Saying “this sort is $O(n^2)$” is true of [merge sort](/cs-glossary/merge-sort/) but not informative; prefer $\Theta(n \log n)$ when you know it.

Drop lower-order terms and constant factors: $3n^2 + 100n = O(n^2)$. Compare algorithms by the dominant term for the $n$ you care about, not only the letter.
