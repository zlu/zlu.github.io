---
title: Place
synonyms:
  - Petri net place
related:
  - Petri net
  - Marking
  - Reachability
  - Mutex
---
A **place** in a [Petri net](/cs-glossary/petri-net/) is a node that holds tokens. Places represent local conditions, buffers, or resource pools; transitions are the events that move tokens between them.

The number of tokens in each place is the [marking](/cs-glossary/marking/). A place with capacity 1 and a single token is the usual drawing of a [mutex](/cs-glossary/mutex/): enter consumes it, exit returns it.

Edges never connect place-to-place or transition-to-transition. The bipartite rule keeps “where tokens sit” separate from “what can fire.”
