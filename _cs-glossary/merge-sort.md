---
title: Merge sort
synonyms:
  - Mergesort
related:
  - Sorting
  - Divide and conquer
  - Quicksort
  - Stable sort
  - Comparison sort
  - Timsort
---
**Merge sort** is a [divide-and-conquer](/cs-glossary/divide-and-conquer/) [comparison sort](/cs-glossary/comparison-sort/): split the array in half, sort each half recursively, then merge two sorted runs into one.

[Time](/cs-glossary/time-complexity/) is $\Theta(n \log n)$ in all cases. Classic implementations need $\Theta(n)$ auxiliary space for the merge buffer. The merge of equal keys can keep left-before-right order, so merge sort is naturally [stable](/cs-glossary/stable-sort/).

It is the conceptual core of [timsort](/cs-glossary/timsort/) and of external sorting (merge runs from disk). Versus [quicksort](/cs-glossary/quicksort/): worse constants and extra memory in the array case, but predictable time and stability.
