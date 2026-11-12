---
title: Sorting
synonyms:
  - Sort
  - Ordering
related:
  - Merge sort
  - Quicksort
  - Heapsort
  - Insertion sort
  - Bubble sort
  - Selection sort
  - Counting sort
  - Radix sort
  - Stable sort
  - Comparison sort
  - Time complexity
---
**Sorting** rearranges a sequence into nondecreasing (or nonincreasing) order under a total order on the keys. Most textbook sorts are [comparison sorts](/cs-glossary/comparison-sort/); [counting sort](/cs-glossary/counting-sort/) and [radix sort](/cs-glossary/radix-sort/) use key structure instead.

Coder view: pick by $n$, key type, and whether you need [stability](/cs-glossary/stable-sort/). Libraries usually ship a hybrid ([timsort](/cs-glossary/timsort/), introsort): [merge](/cs-glossary/merge-sort/)-like for nearly sorted runs, [quick](/cs-glossary/quicksort/)-like for random arrays, heap fallback for worst cases.

Asymptotics you will see on exams: comparison sorts need $\Omega(n \log n)$ comparisons in the worst case in the decision-tree model. Practical $O(n \log n)$ sorts are [merge sort](/cs-glossary/merge-sort/), [heapsort](/cs-glossary/heapsort/), and average-case [quicksort](/cs-glossary/quicksort/). Quadratic $O(n^2)$ sorts ([insertion](/cs-glossary/insertion-sort/), [selection](/cs-glossary/selection-sort/), [bubble](/cs-glossary/bubble-sort/)) still matter for tiny $n$ and for teaching invariants.
