---
title: Amortized analysis
synonyms:
  - Amortized cost
  - Amortized complexity
related:
  - Time complexity
  - Dynamic array
  - Hash table
  - Big-O
---
**Amortized analysis** averages the cost of a sequence of operations, not one worst-case call in isolation. A single `append` that resizes a [dynamic array](/cs-glossary/dynamic-array/) is $O(n)$, but the amortized cost per append is $O(1)$ over a long sequence.

Aggregate, accounting, and potential methods are the three textbook proof styles. Same idea appears in [hash table](/cs-glossary/hash-table/) rehashing and in splay trees.

Amortized $O(1)$ is not the same as worst-case $O(1)$. Soft real-time code that cannot tolerate a rare resize needs a different structure or incremental resizing.
