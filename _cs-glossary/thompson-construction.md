---
title: Thompson construction
related:
  - Regular expression
  - ε-NFA
  - ε-closure
  - NFA
  - Subset construction
  - Kleene star
---
**Thompson construction** builds an [ε-NFA](/cs-glossary/epsilon-nfa/) from a [regular expression](/cs-glossary/regular-expression/) by walking the expression tree and wiring small gadgets for each operator.

Each piece has one entrance and one exit. Union, concatenation, and [Kleene star](/cs-glossary/kleene-star/) always connect with ε-edges the same way. The result is usually fatter than a hand-drawn NFA; ε is glue so the translation is mechanical and easy to prove correct.

After Thompson you remove ε-moves ([ε-closure](/cs-glossary/epsilon-closure/) → ordinary [NFA](/cs-glossary/nfa/)), then optionally [subset-construct](/cs-glossary/subset-construction/) a DFA. That is the standard regex → automaton pipeline.
