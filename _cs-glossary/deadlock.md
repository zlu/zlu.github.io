---
title: Deadlock
related:
  - Mutex
  - Semaphore
  - Critical section
  - Race condition
  - Reachability
  - Petri net
---
**Deadlock** is a permanent wait cycle: each party holds something another needs, and nobody can proceed.

The usual ingredients: mutual exclusion, hold-and-wait, no preemption, and a circular wait. Two threads locking mutexes A then B vs B then A is the textbook. Fix with a global lock order, timeouts, or try-lock with backoff.

In a [Petri net](/cs-glossary/petri-net/), deadlock is a reachable [marking](/cs-glossary/marking/) where no transition is enabled though the system still “should” do work. [Reachability](/cs-glossary/reachability/) analysis is one way to hunt those markings before they hit production.
