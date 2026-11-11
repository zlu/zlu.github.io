---
title: Marking
synonyms:
  - Token marking
related:
  - Petri net
  - Place
  - Reachability
---
A **marking** of a [Petri net](/cs-glossary/petri-net/) is an assignment of a nonnegative token count to every [place](/cs-glossary/place/). It is the global state of the net.

Firing a transition replaces the current marking with a new one: subtract tokens from input places, add tokens to output places. The initial marking $M_0$ is where analysis starts.

[Reachability](/cs-glossary/reachability/) is the question: which markings can appear after a finite firing sequence from $M_0$? Safety and deadlock properties are statements about that set.
