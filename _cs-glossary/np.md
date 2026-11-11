---
title: NP
synonyms:
  - Nondeterministic Polynomial Time
related:
  - P
  - NP-Complete
  - NP-Hard
  - Decision problem
  - Polynomial time
  - Reduction
---
**NP** (nondeterministic polynomial time) is the class of [decision problems](/cs-glossary/decision-problem/) whose yes-instances have a certificate that can be checked in [polynomial time](/cs-glossary/polynomial-time/).

Membership is about verification, not about finding the answer. Sudoku is the usual picture: a filled grid is easy to check; searching for one may be hard. [P](/cs-glossary/p/) sits inside NP: if you can solve it in poly time, you can also verify a yes answer in poly time (ignore the certificate and just solve).

The hard edge of NP is the [NP-complete](/cs-glossary/np-complete/) problems. Every problem in NP [reduces](/cs-glossary/reduction/) to any one of them in polynomial time, so a poly-time algorithm for one would put all of NP in P.
