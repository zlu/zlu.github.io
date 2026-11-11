---
title: Race condition
synonyms:
  - Race
related:
  - Critical section
  - Mutex
  - Semaphore
  - Deadlock
---
A **race condition** means correctness depends on the timing or interleaving of concurrent threads (or processes, or interrupt handlers). Same inputs, different schedules, different results or crashes.

Classic bug: read-modify-write on a shared counter without a [mutex](/cs-glossary/mutex/). Two threads both read 7, both write 8; one increment is lost. Fix by putting the shared access in a [critical section](/cs-glossary/critical-section/), or by using an atomic operation built for that update.

Races are about *unsynchronized conflict*. [Deadlock](/cs-glossary/deadlock/) is the opposite failure mode: too much waiting. Tools (TSAN, lock discipline, immutability) catch many races that code review misses.
