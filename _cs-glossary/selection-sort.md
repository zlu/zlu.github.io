---
title: Selection sort
related:
  - Sorting
  - Bubble sort
  - Insertion sort
  - Heapsort
  - Comparison sort
---
**Selection sort** repeatedly selects the minimum of the unsorted suffix and swaps it into the next prefix position. After $i$ steps, the first $i$ elements are the $i$ smallest in order.

Always $\Theta(n^2)$ comparisons (about $n(n-1)/2$), so no early exit for sorted input. Swaps are few ($O(n)$), which once mattered for expensive writes. Not [stable](/cs-glossary/stable-sort/) in the usual in-place form.

[Heapsort](/cs-glossary/heapsort/) is the same “select next extreme” idea with a [heap](/cs-glossary/heap/) so selection is $O(\log n)$ per step. For teaching, selection sort is the clear $O(n^2)$ baseline; for code, use a library sort.
