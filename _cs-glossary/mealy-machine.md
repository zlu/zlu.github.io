---
title: Mealy machine
related:
  - Moore machine
  - Transducer
  - Automaton
  - Alphabet
---
A **Mealy machine** is a finite [transducer](/cs-glossary/transducer/) whose output depends on the *transition*: state and input together, $\lambda: Q \times \Sigma \to \Gamma$.

Diagrams label arcs with `input / output`. Output changes on the edge, so an $n$-symbol input yields $n$ outputs (no free start-state tick the way Moore has). Reaction can be immediate on the event that caused the move.

Same controller briefs can often be drawn as Moore or Mealy; the difference is when the command appears relative to the input. Pick the style that matches your timing requirements, then keep the drawing consistent.
