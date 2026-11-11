---
title: P
synonyms:
  - Polynomial Time
  - Deterministic Polynomial Time
related:
  - NP
  - Polynomial time
  - Decision problem
  - NP-Complete
---
**P** is the class of [decision problems](/cs-glossary/decision-problem/) solvable by a deterministic Turing machine in [polynomial time](/cs-glossary/polynomial-time/): time $O(n^k)$ for some fixed $k$ in the input size $n$.

Sorting-related decisions, shortest paths in graphs with nonnegative weights, and 2-SAT sit in P. These are the problems theory treats as “tractable” in the worst-case asymptotic sense.

P $\subseteq$ [NP](/cs-glossary/np/). Whether equality holds is open. [NP-complete](/cs-glossary/np-complete/) problems are the candidates that would collapse the two classes if any one of them fell into P.
