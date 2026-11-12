---
title: Divide and conquer
synonyms:
  - Divide-and-conquer
related:
  - Merge sort
  - Quicksort
  - Binary search
  - Recursion
  - Dynamic programming
  - Master theorem
---
**Divide and conquer** solves a problem by splitting into independent subproblems, solving them recursively, then combining results. Classic template: divide, conquer, combine.

[Merge sort](/cs-glossary/merge-sort/), [quicksort](/cs-glossary/quicksort/), and [binary search](/cs-glossary/binary-search/) are the usual examples. Recurrences such as $T(n) = 2T(n/2) + O(n)$ are solved with the [master theorem](/cs-glossary/master-theorem/) or recursion trees.

Unlike [dynamic programming](/cs-glossary/dynamic-programming/), pure D&C does not overlap subproblems; if they overlap and you recompute, switch to DP or memoization. Unlike [greedy](/cs-glossary/greedy-algorithm/) methods, D&C explores whole subtrees of the recursion, not one local choice.
