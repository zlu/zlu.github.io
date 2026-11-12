---
title: Stable sort
synonyms:
  - Stability
  - Sorting stability
related:
  - Sorting
  - Merge sort
  - Insertion sort
  - Counting sort
  - Timsort
  - Quicksort
---
A **stable sort** preserves the relative order of equal keys. If two records share a key and $A$ appeared before $B$ in the input, $A$ still appears before $B$ after sorting.

Stability matters when you sort on one field, then another: a stable secondary sort keeps the primary order among ties. [Merge sort](/cs-glossary/merge-sort/), [insertion sort](/cs-glossary/insertion-sort/), [counting sort](/cs-glossary/counting-sort/), and [timsort](/cs-glossary/timsort/) are typically stable. Naive [quicksort](/cs-glossary/quicksort/) and [heapsort](/cs-glossary/heapsort/) are not.

In APIs, check the docs: `Collections.sort` / `Arrays.sort` for objects is often a stable timsort; primitive sorts may trade stability for speed.
