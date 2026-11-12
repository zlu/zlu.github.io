---
title: Insertion sort
related:
  - Sorting
  - Bubble sort
  - Merge sort
  - Stable sort
  - Comparison sort
---
**Insertion sort** builds a sorted prefix one element at a time: take $a[i]$, shift larger prefix elements right, and drop it into place. Same idea as sorting playing cards in your hand.

Worst and average [time](/cs-glossary/time-complexity/) $O(n^2)$; best case $O(n)$ on already sorted input. [Stable](/cs-glossary/stable-sort/), in-place, excellent constant factors on tiny arrays. [Timsort](/cs-glossary/timsort/) and many introsorts use it for short runs.

Prefer it for $n$ up to a few dozen, nearly sorted data, or online insertion into a sorted buffer. For large random arrays use [merge](/cs-glossary/merge-sort/) / [quick](/cs-glossary/quicksort/) / library sort.
