---
title: Decision problem
synonyms:
  - Yes/no problem
related:
  - P
  - NP
  - NP-Complete
  - Reduction
---
A **decision problem** is a yes/no question on a finite encoding of the input: “does this graph have a Hamiltonian cycle?” not “find one.”

Complexity classes like [P](/cs-glossary/p/) and [NP](/cs-glossary/np/) are defined on decision problems. Optimization (“find the shortest tour”) is often cast as a family of decisions (“is there a tour of length $\leq k$?”) so membership and [reductions](/cs-glossary/reduction/) apply cleanly.

[NP-complete](/cs-glossary/np-complete/) status is about decision versions. The search or optimization sibling may be [NP-hard](/cs-glossary/np-hard/) without itself being a decision problem in NP.
