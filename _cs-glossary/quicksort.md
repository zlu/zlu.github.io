---
title: Quicksort
synonyms:
  - Quick sort
  - Partition sort
related:
  - Sorting
  - Divide and conquer
  - Merge sort
  - Heapsort
  - Comparison sort
  - Stable sort
---
**Quicksort** picks a pivot, [partitions](/cs-glossary/partition/) so smaller keys go left and larger go right, then recursively sorts the two sides. It is [divide and conquer](/cs-glossary/divide-and-conquer/) without a merge step; the work is in the partition.

Average [time](/cs-glossary/time-complexity/) $O(n \log n)$ with excellent locality; worst case $O(n^2)$ on bad pivots (already sorted with a naive first/last pivot). Randomized or median-of-three pivots, plus a heap-sort fallback (introsort), make worst-case rare in practice. Typical in-place forms are not [stable](/cs-glossary/stable-sort/).

Default choice inside many language runtimes for primitives. For guaranteed $O(n \log n)$ and stability, prefer [merge sort](/cs-glossary/merge-sort/) / [timsort](/cs-glossary/timsort/).
