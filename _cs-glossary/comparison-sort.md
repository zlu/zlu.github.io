---
title: Comparison sort
synonyms:
  - Comparison-based sort
related:
  - Sorting
  - Merge sort
  - Quicksort
  - Heapsort
  - Insertion sort
  - Lower bound
---
A **comparison sort** orders elements using only pairwise comparisons ($a < b$, $a \le b$, …). It does not inspect bit patterns or digit ranks the way [counting](/cs-glossary/counting-sort/) or [radix](/cs-glossary/radix-sort/) sorts do.

In the decision-tree model, any comparison sort needs $\Omega(n \log n)$ comparisons in the worst case to sort $n$ distinct keys. That is why “better than $n \log n$” for arbitrary comparable keys usually means a different model (integers in a range, strings with radix structure), not a smarter compare.

[Merge sort](/cs-glossary/merge-sort/), [heapsort](/cs-glossary/heapsort/), [quicksort](/cs-glossary/quicksort/), and the $O(n^2)$ elementary sorts are all comparison sorts.
