---
title: Priority queue
synonyms:
  - PQ
related:
  - Heap
  - Queue
  - Dijkstra
  - Huffman coding
  - Heapsort
---
A **priority queue** stores elements with keys and always supports “extract the best key” (min or max), plus insert. A binary [heap](/cs-glossary/heap/) is the usual implementation; Fibonacci heaps appear in theory for better amortized decrease-key.

[Dijkstra](/cs-glossary/dijkstra/), A*, event simulators, and [Huffman coding](/cs-glossary/huffman-coding/) are consumers. It is not a FIFO [queue](/cs-glossary/queue/): order is by priority, not arrival.

Operations are typically $O(\log n)$ per insert/extract with a binary heap; peek is $O(1)$.
