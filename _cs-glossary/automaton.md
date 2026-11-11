---
title: Automaton
synonyms:
  - Automata
  - State machine
  - Accepter
related:
  - DFA
  - NFA
  - Transducer
  - Formal language
  - Accepting state
  - Petri net
---
An **automaton** is a state machine that reads input and either answers accept/reject (**accepter**) or emits outputs as it runs (**[transducer](/cs-glossary/transducer/)**).

Finite accepters ([DFA](/cs-glossary/dfa/), [NFA](/cs-glossary/nfa/)) are the validators of theory: same job as `isValid…()` or a regex engine’s match. Start state, transitions on symbols, [accepting states](/cs-glossary/accepting-state/). The language is the set of strings that end accepting.

Beyond finite accepters you get pushdown automata, Turing machines, and models like [Petri nets](/cs-glossary/petri-net/) for concurrent token flow. Same instinct: finite description, precise runs, yes/no or observable behavior.
