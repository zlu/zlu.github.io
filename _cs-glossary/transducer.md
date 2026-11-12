---
title: Transducer
synonyms:
  - Transducers
  - Finite transducer
  - Finite transducers
related:
  - Moore machine
  - Mealy machine
  - Automaton
  - Alphabet
  - DFA
  - Formal language
---
A **transducer** is an [automaton](/cs-glossary/automaton/) that produces an *output stream* while it runs, not just accept/reject at the end.

Finite transducers still have states, an input [alphabet](/cs-glossary/alphabet/) $\Sigma$, and transitions. They also have an output alphabet $\Gamma$. [Moore](/cs-glossary/moore-machine/) attaches output to states; [Mealy](/cs-glossary/mealy-machine/) attaches it to transitions.

Accepters ([DFAs](/cs-glossary/dfa/), [NFAs](/cs-glossary/nfa/)) answer language membership. Transducers produce outputs while reading input: $\Sigma$ is events, $\Gamma$ is commands (motors, lamps, UI modes). Same finite-state setup, different job.
