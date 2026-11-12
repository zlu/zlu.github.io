---
title: Linked list
synonyms:
  - Singly linked list
  - Doubly linked list
related:
  - Array
  - Stack
  - Queue
  - Dynamic array
---
A **linked list** stores elements in nodes with pointers to neighbors. Singly linked lists have a `next`; doubly linked lists also have `prev`. Access by index is $O(n)$; insert/delete at a known node is $O(1)$ pointer updates.

Good for [stacks](/cs-glossary/stack/), [queues](/cs-glossary/queue/), and intrusive lists in systems code. Poor spatial locality versus [arrays](/cs-glossary/array/). Binary search does not apply.

In high-level application code, a [dynamic array](/cs-glossary/dynamic-array/) is usually faster unless you truly need middle splices without shifting.
