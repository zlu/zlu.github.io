---
title: Recurrence
synonyms:
  - Recurrence relation
related:
  - Master theorem
  - Divide and conquer
  - Dynamic programming
  - Time complexity
---
A **recurrence** defines $T(n)$ in terms of smaller values, usually the running time of a recursive algorithm: e.g. $T(n) = 2T(n/2) + O(n)$ for [merge sort](/cs-glossary/merge-sort/).

Solve with the [master theorem](/cs-glossary/master-theorem/), a recursion tree, or substitution (guess and induct). [DP](/cs-glossary/dynamic-programming/) also uses recurrences, but for *values* of subproblems, not only for runtime.

Always include base cases. Floors, ceilings, and uneven splits rarely change $\Theta$ class, but they can break a sloppy closed form.
