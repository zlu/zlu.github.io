---
title: Heapsort
synonyms:
  - Heap sort
related:
  - Sorting
  - Heap
  - Priority queue
  - Selection sort
  - Comparison sort
  - Quicksort
---
**Heapsort** builds a max-[heap](/cs-glossary/heap/) on the array, then repeatedly swaps the root (current maximum) with the last unsorted position and sifts down. It is [selection sort](/cs-glossary/selection-sort/) with logarithmic selection.

[Time](/cs-glossary/time-complexity/) $\Theta(n \log n)$ worst case, $O(1)$ extra memory beyond the array. Not [stable](/cs-glossary/stable-sort/). Cache behavior is usually worse than [quicksort](/cs-glossary/quicksort/), so libraries often use heapify only as a worst-case fallback (introsort).

Good exam answer when asked for an in-place comparison sort with guaranteed $O(n \log n)$.
