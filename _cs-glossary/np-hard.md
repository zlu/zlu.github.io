---
title: NP-Hard
related:
  - NP
  - NP-Complete
  - P
  - Reduction
  - Decision problem
---
**NP-hard** problems are at least as hard as every problem in [NP](/cs-glossary/np/). A [polynomial-time](/cs-glossary/polynomial-time/) algorithm for any NP-hard problem would yield one for every problem in NP.

Unlike [NP-complete](/cs-glossary/np-complete/) problems, an NP-hard problem need not be in NP. Optimization versions (find the shortest tour, not just decide whether a short tour exists) are often NP-hard without being NP-complete, because they are not [decision problems](/cs-glossary/decision-problem/) with a short yes-certificate in the NP sense.

Problem $P$ is NP-hard if every problem in NP [reduces](/cs-glossary/reduction/) to $P$ in polynomial time. In practice, scheduling, routing, and combinatorial search tasks that are NP-hard get heuristics, approximation algorithms, or exact solvers that work on typical instances even when worst-case poly time is unavailable.
