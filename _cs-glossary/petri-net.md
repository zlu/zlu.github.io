---
title: Petri net
related:
  - Place
  - Marking
  - Reachability
  - Mutex
  - Semaphore
  - Automaton
---
A **Petri net** is a bipartite graph of [places](/cs-glossary/place/) and transitions, with tokens that move when transitions fire. It models concurrent, resource-aware behavior that a single finite automaton thread does not capture cleanly.

A transition may fire when every input place holds enough tokens; firing consumes those tokens and produces tokens on output places. The [marking](/cs-glossary/marking/) (token count per place) is the global state. [Reachability](/cs-glossary/reachability/) asks which markings you can get to from the initial one.

Classic concurrency patterns sit naturally here: a [mutex](/cs-glossary/mutex/) is one token in a resource place; a [semaphore](/cs-glossary/semaphore/) is a place with capacity $k$. Deadlock shows up as no enabled transition when you still expected progress.
