---
title: Heap
synonyms:
  - Binary heap
  - Max-heap
  - Min-heap
related:
  - Priority queue
  - Heapsort
  - Dijkstra
  - Array
  - Huffman coding
---
A **heap** (usually a binary heap) is a nearly complete [binary tree](/cs-glossary/binary-tree/) stored in an [array](/cs-glossary/array/) that satisfies the heap order: parent $\le$ children (min-heap) or $\ge$ (max-heap).

`insert` and `extract-min` / `extract-max` are $O(\log n)$; peek is $O(1)$; build-heap is $O(n)$. It is the default [priority queue](/cs-glossary/priority-queue/) in courses and the structure behind [heapsort](/cs-glossary/heapsort/), [Dijkstra](/cs-glossary/dijkstra/), and [Huffman](/cs-glossary/huffman-coding/).

Not sorted end-to-end: only the root is guaranteed extreme. For ordered iteration use a balanced BST or sort explicitly.
