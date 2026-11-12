---
title: Deque
synonyms:
  - Double-ended queue
related:
  - Queue
  - Stack
  - Dynamic array
---
A **deque** (double-ended queue) supports push/pop at both front and back in $O(1)$ time. It generalizes [stack](/cs-glossary/stack/) and [queue](/cs-glossary/queue/) operations in one structure.

Python `collections.deque`, C++ `std::deque`, and ring-buffer designs are typical. Sliding-window maximum and BFS variants that need both ends use deques.
