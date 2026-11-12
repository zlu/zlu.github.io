---
title: Space complexity
synonyms:
  - Memory complexity
  - Auxiliary space
related:
  - Time complexity
  - Big-O
  - Recursion
  - In-place
---
**Space complexity** measures memory as a function of input size. Distinguish total space from *auxiliary* space (extra beyond the input). An [in-place](/cs-glossary/in-place/) [heapsort](/cs-glossary/heapsort/) uses $O(1)$ auxiliary words; [merge sort](/cs-glossary/merge-sort/) typically needs $\Theta(n)$ auxiliary.

[Recursion](/cs-glossary/recursion/) depth counts: a naive recursive DFS can use $O(V)$ stack frames. Streaming algorithms aim for sublinear extra space.

Trade space for time when tables, caches, or memoization ([DP](/cs-glossary/dynamic-programming/)) remove repeated work.
