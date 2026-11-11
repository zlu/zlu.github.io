---
title: Reachability
related:
  - Petri net
  - Marking
  - Place
  - Deadlock
---
**Reachability** asks whether a target [marking](/cs-glossary/marking/) (or class of markings) can appear after finitely many firings from the initial marking of a [Petri net](/cs-glossary/petri-net/).

Build the reachability graph when it is finite: nodes are markings, edges are firings. Coverability graphs handle unbounded nets. Many safety properties reduce to “bad marking unreachable.”

In concurrency terms: can this resource allocation, buffer fill, or deadlock [marking](/cs-glossary/marking/) actually happen? If nothing is enabled and you still needed progress, that reachable dead marking is [deadlock](/cs-glossary/deadlock/).
