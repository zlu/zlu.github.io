---
title: Radix sort
related:
  - Sorting
  - Counting sort
  - Stable sort
  - Comparison sort
---
**Radix sort** sorts keys digit by digit (or byte by byte), usually least-significant digit first (LSD), with a [stable](/cs-glossary/stable-sort/) pass such as [counting sort](/cs-glossary/counting-sort/) on each digit.

For $n$ keys of $d$ digits in base $r$, time is $O(d(n + r))$. Fixed-width integers and strings with bounded length are the sweet spot. It is not a [comparison sort](/cs-glossary/comparison-sort/).

MSD radix sort can short-circuit on prefixes and is closer to trie / string algorithms. For general `Comparable` objects, stick to comparison sorts.
