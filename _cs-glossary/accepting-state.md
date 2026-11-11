---
title: Accepting state
synonyms:
  - Final state
  - Accept state
related:
  - DFA
  - NFA
  - Automaton
  - Formal language
---
An **accepting state** (final state) is a state in $F$ where an accepter says yes if the input ends there.

For a [DFA](/cs-glossary/dfa/), run $\delta^*$ from the start; accept iff the unique end state is in $F$. For an [NFA](/cs-glossary/nfa/), accept iff *some* run ends in $F$. Diagrams usually mark finals with a double circle.

The [language](/cs-glossary/formal-language/) of the machine is exactly the strings whose runs finish accepting. No $F$ means transducers or pure controllers: output matters, not membership.
