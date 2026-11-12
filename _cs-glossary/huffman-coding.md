---
title: Huffman coding
synonyms:
  - Huffman code
  - Huffman tree
related:
  - Greedy algorithm
  - Priority queue
  - Heap
  - Binary tree
---
**Huffman coding** builds an optimal prefix-free binary code for known symbol frequencies: repeatedly merge the two lightest trees, using a [priority queue](/cs-glossary/priority-queue/) / [heap](/cs-glossary/heap/). It is the standard [greedy](/cs-glossary/greedy-algorithm/) example in algorithms courses.

The resulting code tree assigns short bitstrings to frequent symbols. Optimality is among prefix codes for those frequencies; it is not magic compression for arbitrary files without a frequency model.

Time $O(n \log n)$ for $n$ symbols with a binary heap. Related ideas show up in JPEG-style pipelines and in teaching priority queues.
