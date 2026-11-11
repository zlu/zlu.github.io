---
title: DFA
synonyms:
  - Deterministic Finite Automaton
  - Deterministic Finite Accepter
related:
  - NFA
  - ε-NFA
  - Accepting state
  - Alphabet
  - Regular language
  - Subset construction
  - Automaton
---
A **DFA** (deterministic finite automaton / accepter) is a finite-state machine that, from every state and every [alphabet](/cs-glossary/alphabet/) symbol, has exactly one next state. No guesswork, no ε-moves.

Formally $M = (Q, \Sigma, \delta, q_0, F)$ with total $\delta: Q \times \Sigma \to Q$. Run a string left to right from $q_0$; accept iff you land in an [accepting state](/cs-glossary/accepting-state/) in $F$. That is the same job as a hard-coded validator or a compiled regex matcher after [subset construction](/cs-glossary/subset-construction/).

DFAs and [NFAs](/cs-glossary/nfa/) describe exactly the [regular languages](/cs-glossary/regular-language/). The DFA may need exponentially many states versus an equivalent NFA, but membership is then a single deterministic walk: one state, one symbol, one step.
