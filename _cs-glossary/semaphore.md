---
title: Semaphore
related:
  - Mutex
  - Critical section
  - Deadlock
  - Race condition
  - Petri net
---
A **semaphore** is a synchronization counter with `wait`/`P` (decrement, block if zero) and `signal`/`V` (increment, wake a waiter). It limits how many threads may hold a resource at once.

A binary semaphore (count 0/1) overlaps a [mutex](/cs-glossary/mutex/), though mutexes usually add ownership and recursive-lock rules. Counting semaphores gate pools: three identical buffers, three permits.

In a [Petri net](/cs-glossary/petri-net/), the semaphore is a [place](/cs-glossary/place/) whose tokens are permits. Misordered waits across semaphores are a standard [deadlock](/cs-glossary/deadlock/) recipe; forgetting a signal starves everyone behind you.
