---
title: Timsort
related:
  - Sorting
  - Merge sort
  - Insertion sort
  - Stable sort
---
**Timsort** is a [stable](/cs-glossary/stable-sort/) hybrid [sort](/cs-glossary/sorting/) used by CPython and (historically) Java’s object `Arrays.sort`. It finds natural ascending or descending runs, sorts short runs with [insertion sort](/cs-glossary/insertion-sort/), then merges them with a [merge-sort](/cs-glossary/merge-sort/) strategy and carefully chosen run-stack invariants.

On random data it is still $O(n \log n)$. On nearly sorted data it can be close to linear because long runs need little work. That is why “just call the language sort” is usually the right default unless you have a special key domain.
