---
title: Mutex
synonyms:
  - Mutual exclusion
  - Lock
related:
  - Semaphore
  - Critical section
  - Race condition
  - Deadlock
---
A **mutex** (mutual exclusion) is a lock that lets at most one holder use a shared resource at a time. Acquire before the [critical section](/cs-glossary/critical-section/), release after; everyone else waits.

In code this is an OS or library lock (`pthread_mutex`, `std::mutex`, Java `synchronized`). Holding too long hurts throughput. Forgetting to release, or locking in inconsistent order across threads, is a classic path to [deadlock](/cs-glossary/deadlock/).

In a [Petri net](/cs-glossary/petri-net/), the same idea is a single resource token in a [place](/cs-glossary/place/): the enter transition consumes it, the exit transition returns it. A [semaphore](/cs-glossary/semaphore/) generalizes the count; a mutex is the count-1 case with ownership conventions.
