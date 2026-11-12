---
title: In-place
synonyms:
  - In place algorithm
  - Constant extra space
related:
  - Space complexity
  - Heapsort
  - Quicksort
  - Merge sort
---
An **in-place** algorithm transforms the input using only $O(1)$ (or at most $O(\log n)$ for recursion stacks, depending on the course definition) extra memory beyond the input storage.

[Heapsort](/cs-glossary/heapsort/) and typical [quicksort](/cs-glossary/quicksort/) partitions are in-place; classic [merge sort](/cs-glossary/merge-sort/) is not. “In-place” is about [space complexity](/cs-glossary/space-complexity/), not about time.

Check the definition your exam uses: some allow the $O(\log n)$ call stack; others demand strictly constant auxiliary words.
