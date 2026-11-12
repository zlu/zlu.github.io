---
title: Quickselect
synonyms:
  - Hoare's selection algorithm
  - Quick select
related:
  - Quicksort
  - Partition
  - Heap
  - Binary search
---
**Quickselect** finds the $k$-th smallest element (order statistic) using the same [partition](/cs-glossary/partition/) step as [quicksort](/cs-glossary/quicksort/), but recurses into only one side.

Average [time](/cs-glossary/time-complexity/) $O(n)$; worst case $O(n^2)$ with bad pivots. Expected linear time with random pivots is the usual practical claim. A [heap](/cs-glossary/heap/) also solves top-$k$ problems, with different constants and a clearer worst case for the heapify path.

Use when you need one order statistic, not a full [sort](/cs-glossary/sorting/). Median-of-medians gives deterministic linear worst case but is rarely what you ship.
