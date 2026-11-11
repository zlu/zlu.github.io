---
title: NFA
synonyms:
  - Nondeterministic Finite Automaton
  - Nondeterministic Finite Accepter
related:
  - DFA
  - ε-NFA
  - Subset construction
  - Accepting state
  - Regular language
  - Automaton
---
An **NFA** (nondeterministic finite automaton / accepter) allows zero, one, or many next states for the same (state, symbol) pair. Acceptance means *some* run ends in an [accepting state](/cs-glossary/accepting-state/).

Coder view: the machine explores a set of possible states in parallel. On each symbol you take the union of all δ-images of the current set. That set-of-states reading is exactly what [subset construction](/cs-glossary/subset-construction/) turns into a [DFA](/cs-glossary/dfa/) state.

NFAs do not accept more languages than DFAs; they accept the same [regular languages](/cs-glossary/regular-language/), often with a much smaller drawing. An [ε-NFA](/cs-glossary/epsilon-nfa/) adds free jumps; strip those with [ε-closure](/cs-glossary/epsilon-closure/) and you are back to an ordinary NFA.
