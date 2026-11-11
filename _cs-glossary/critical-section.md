---
title: Critical section
synonyms:
  - Critical region
related:
  - Mutex
  - Semaphore
  - Race condition
  - Deadlock
---
A **critical section** is the stretch of code that touches shared state and must not run concurrently with conflicting sections on that state.

You bracket it with a [mutex](/cs-glossary/mutex/) or [semaphore](/cs-glossary/semaphore/): acquire, touch the shared object, release. Keep it short. Long critical sections serialize threads and raise [deadlock](/cs-glossary/deadlock/) risk if you nest locks.

Without exclusion you get [race conditions](/cs-glossary/race-condition/): results depend on interleaving. The Petri-net picture is enter/exit transitions around a resource token; the section is the path while the token is held.
