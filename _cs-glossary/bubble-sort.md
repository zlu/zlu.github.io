---
title: Bubble sort
synonyms:
  - Bubblesort
related:
  - Sorting
  - Insertion sort
  - Selection sort
  - Comparison sort
---
**Bubble sort** repeatedly passes over the array, swapping adjacent out-of-order pairs so larger keys “bubble” toward the end. After pass $i$, the last $i$ positions are in final order.

Worst and average [time](/cs-glossary/time-complexity/) are $O(n^2)$; best case with an early-exit flag is $O(n)$ on a sorted array. It is a [comparison sort](/cs-glossary/comparison-sort/), usually stable if equal keys are not swapped.

Almost never the right production choice. Use it to practice loop invariants and adjacent swaps; prefer library sorts or [insertion sort](/cs-glossary/insertion-sort/) for tiny $n$.
