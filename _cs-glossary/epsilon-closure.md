---
title: ε-closure
synonyms:
  - epsilon-closure
  - λ-closure
  - lambda-closure
related:
  - ε-NFA
  - NFA
  - Subset construction
  - Thompson construction
---
**ε-closure** (Linz: λ-closure) of a state set $S$ is every state reachable from $S$ by following only $\varepsilon$ / $\lambda$ edges, including $S$ itself.

When you simulate an [ε-NFA](/cs-glossary/epsilon-nfa/), after reading a real symbol (and at the start) you replace the current set by its ε-closure. Free jumps do not consume input; they just enlarge where you “are.”

The same operation turns ε-NFAs into ordinary [NFAs](/cs-glossary/nfa/) for [subset construction](/cs-glossary/subset-construction/): fold each ε-path into the state name so remaining edges are on alphabet symbols only.
