---
title: Counting sort
related:
  - Sorting
  - Radix sort
  - Stable sort
  - Comparison sort
---
**Counting sort** sorts integers (or keys mapped to integers) in a known range $\{0,\ldots,k\}$ by counting occurrences, prefix-summing the counts into positions, then placing each record into an output array.

[Time](/cs-glossary/time-complexity/) and space are $O(n + k)$. It is not a [comparison sort](/cs-glossary/comparison-sort/), so it can beat $n \log n$ when $k = O(n)$. Done carefully (scan input backward into the cumulative positions) it is [stable](/cs-glossary/stable-sort/), which [radix sort](/cs-glossary/radix-sort/) needs for digit passes.

Use when keys are small integers or bytes. Skip when $k$ is huge relative to $n$ or keys are arbitrary comparable objects.
