---
title: NP-Complete
related:
  - NP
  - NP-Hard
  - P
  - Reduction
  - Decision problem
  - Polynomial time
---
**NP-complete** problems are the hardest [decision problems](/cs-glossary/decision-problem/) in [NP](/cs-glossary/np/): they are in NP, and every problem in NP [reduces](/cs-glossary/reduction/) to them in [polynomial time](/cs-glossary/polynomial-time/).

If any NP-complete problem has a polynomial-time algorithm, then every problem in NP does ([P](/cs-glossary/p/) = NP). Classic examples: 3-SAT, Hamiltonian cycle, and the decision version of traveling salesman (does a tour of length at most $k$ exist?).

NP-complete problems are also [NP-hard](/cs-glossary/np-hard/). The converse fails when the problem is not known (or not) to be in NP, for example many optimization versions that ask for the best object rather than a yes/no answer.
