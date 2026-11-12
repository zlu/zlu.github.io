---
title: Master theorem
synonyms:
  - Master method
related:
  - Divide and conquer
  - Merge sort
  - Recurrence
  - Time complexity
---
The **master theorem** gives asymptotic solutions for many [divide-and-conquer](/cs-glossary/divide-and-conquer/) recurrences of the form $T(n) = a T(n/b) + f(n)$ with constants $a \ge 1$, $b > 1$. Compare $f(n)$ to $n^{\log_b a}$ (the critical exponent) to pick the case.

Useful defaults: [merge sort](/cs-glossary/merge-sort/) $T(n) = 2T(n/2) + \Theta(n)$ is $\Theta(n \log n)$. Binary-search-like $T(n) = T(n/2) + \Theta(1)$ is $\Theta(\log n)$.

Not every recurrence fits (uneven splits, floors/ceilings that matter, non-polynomial differences). Then use a recursion tree or substitution proof instead.
