---
title: Queue
synonyms:
  - FIFO
related:
  - Stack
  - BFS
  - Deque
  - Priority queue
  - Linked list
---
A **queue** is a FIFO (first-in, first-out) collection: enqueue at the back, dequeue at the front. Ring buffers ([arrays](/cs-glossary/array/)) and [linked lists](/cs-glossary/linked-list/) are common implementations.

[BFS](/cs-glossary/bfs/) is the algorithms use case: the queue holds the frontier by discovery order. Task scheduling and IO buffers use the same idea.

Core operations are $O(1)$. A [priority queue](/cs-glossary/priority-queue/) is not FIFO; it always extracts the best key instead.
