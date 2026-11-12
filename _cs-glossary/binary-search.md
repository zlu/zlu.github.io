---
title: Binary search
synonyms:
  - Bisection search
  - Logarithmic search
related:
  - Searching
  - Linear search
  - Array
  - Binary search tree
  - Divide and conquer
---
**Binary search** finds a target in a sorted [array](/cs-glossary/array/) by comparing against the middle element and discarding half the remaining range each step. The search interval is always a contiguous sorted slice.

[Time](/cs-glossary/time-complexity/) $O(\log n)$ comparisons; $O(1)$ extra space for the iterative form. Off-by-one errors on `lo` / `hi` / midpoint are the classic bug. Variants return the leftmost or rightmost insertion point for duplicates (lower_bound / upper_bound).

Requires random access and a sorted order. Linked lists do not get the asymptotic win. A [binary search tree](/cs-glossary/binary-search-tree/) stores the same compare-and-discard idea in pointer form.
