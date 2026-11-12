---
title: Partition
synonyms:
  - Array partition
  - Pivot partition
related:
  - Quicksort
  - Quickselect
  - Binary search
---
**Partition** rearranges an array around a pivot value so every element before an index $i$ is $\le$ pivot and every element after is $\ge$ pivot (scheme variants differ on equals). Hoare and Lomuto are the two common in-place recipes taught with [quicksort](/cs-glossary/quicksort/).

Partition is also the core of [quickselect](/cs-glossary/quickselect/) for order statistics: after one partition you know which side holds the $k$-th element and recurse only there.

Cost is $\Theta(n)$ for a contiguous segment of length $n$. Correctness hinges on the invariant you maintain while scanning; off-by-one bugs here are the usual quicksort failure mode.
