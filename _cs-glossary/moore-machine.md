---
title: Moore machine
related:
  - Mealy machine
  - Transducer
  - Automaton
  - Alphabet
---
A **Moore machine** is a finite [transducer](/cs-glossary/transducer/) whose output is a function of the *current state* alone: $\lambda: Q \to \Gamma$. While you sit in $q$, the output is $\lambda(q)$.

Arcs carry input only; the command lives inside the bubble. Entering a state “turns on” its output until you leave. On an $n$-symbol input you get $n+1$ outputs (one for the start state before any input, then one after each step).

Controllers map cleanly to Moore: states are modes (lamp level, light phase), inputs are events, outputs are commands to hardware or UI. [Mealy machines](/cs-glossary/mealy-machine/) put output on the transition instead; same jobs are often convertible with a timing shift.
