---
title: ε-NFA
synonyms:
  - λ-NFA
  - epsilon-NFA
  - lambda-NFA
related:
  - NFA
  - DFA
  - ε-closure
  - Thompson construction
  - Regular expression
  - Subset construction
---
An **ε-NFA** (Linz: **λ-NFA**) is an [NFA](/cs-glossary/nfa/) that may also move on the empty string: edges labeled $\varepsilon$ / $\lambda$ that read no input.

Those free jumps are glue. [Thompson construction](/cs-glossary/thompson-construction/) invents them so every regex operator wires the same way (one entrance, one exit). They are not a deeper feature of [regular languages](/cs-glossary/regular-language/); you delete them later.

To run or to convert: take [ε-closure](/cs-glossary/epsilon-closure/) after every real symbol (and at the start). Collapse ε-paths into ordinary letter edges to get an NFA, then optionally [subset-construct](/cs-glossary/subset-construction/) a [DFA](/cs-glossary/dfa/).
