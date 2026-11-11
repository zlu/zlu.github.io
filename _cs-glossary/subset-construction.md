---
title: Subset construction
synonyms:
  - Powersets construction
  - NFA to DFA
related:
  - NFA
  - DFA
  - ε-closure
  - ε-NFA
  - Regular language
---
**Subset construction** converts an [NFA](/cs-glossary/nfa/) into an equivalent [DFA](/cs-glossary/dfa/). Each DFA state is a *set* of NFA states: the set of places you might be after reading the same prefix.

Start from $\{q_0\}$ (or [ε-closure](/cs-glossary/epsilon-closure/) of the start, if you have an [ε-NFA](/cs-glossary/epsilon-nfa/)). From a subset $S$, on symbol $a$, go to the set of all NFA successors of states in $S$ (then ε-close again if needed). A DFA state is accepting if any member is accepting.

Lazy construction only builds subsets you actually reach; you rarely materialize the full powerset. Worst case the DFA is exponential in the NFA size, but the language is unchanged: still [regular](/cs-glossary/regular-language/).
