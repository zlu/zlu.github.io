---
title: FRP
synonyms:
  - Functional Reactive Programming
  - Functional Reactive Animation
related:
  - Haskell
  - React
  - Transducer
  - Smalltalk
---
**FRP** (functional reactive programming) treats values that vary over time as first-class data. Classic FRP (Fran, Yampa, Reactive Banana, Sodium) has two primitives:

- **Event a**: discrete occurrences over time, each carrying a value of type `a`
- **Behavior a**: a continuous value of type `a` defined at every time $t$

You build new Events and Behaviors with combinators (`map`, `filter`, `merge`, `hold`, `tag` / `snapshot`, `accumB`). Rx-style `Observable` libraries are closer to Events only; they do not ship a true continuous Behavior with the same sampling semantics.

Fran (Elliott and Hudak, ICFP 1997) implemented the early model in [Haskell](/cs-glossary/haskell/).
