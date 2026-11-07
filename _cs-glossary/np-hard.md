---
title: NP-Hard
related:
  - NP
  - NP-Complete
---
**NP-hard** problems are at least as hard as every problem in [NP](/cs-glossary/np/). A polynomial-time algorithm for any NP-hard problem would yield a polynomial-time algorithm for every problem in NP.

Unlike [NP-complete](/cs-glossary/np-complete/) problems, an NP-hard problem need not be in NP. Optimization versions (find the shortest tour, not just decide whether a short tour exists) are often NP-hard without being NP-complete, because they are not decision problems with a short yes-certificate in the NP sense.

### Reduction

Problem $P$ is NP-hard if every problem in NP reduces to $P$ in polynomial time. Solving $P$ would solve all of NP.

### Examples

- Traveling salesman (optimization): shortest tour visiting each city once and returning home
- Knapsack (optimization): highest-value subset of items under a weight limit
- Graph coloring (optimization): fewest colors so adjacent vertices differ
- 3-SAT: NP-complete, hence also NP-hard

### Why it matters

Many scheduling, routing, and combinatorial search tasks are NP-hard. In practice people use heuristics, approximation algorithms, or exact solvers that work on typical instances even when worst-case polynomial time is unavailable.
