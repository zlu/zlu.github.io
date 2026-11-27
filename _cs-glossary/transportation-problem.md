---
title: Transportation problem
synonyms:
  - Hitchcock transportation problem
  - Transport problem
keywords: transportation problem, supply demand, bipartite flow, OR, optimisation
related:
  - Transshipment problem
  - Minimum cost flow
  - Linear programming
  - Balanced transportation
---
The **transportation problem** ships a homogeneous good from supply nodes to demand nodes at minimum cost:

$$
\min\sum_{i,j}c_{ij}x_{ij}
\quad\text{s.t.}\quad
\sum_j x_{ij}=s_i,\
\sum_i x_{ij}=d_j,\
x_{ij}\ge 0.
$$

Special [LP](/cs-glossary/linear-programming/)/network structure. [Transshipment](/cs-glossary/transshipment-problem/) adds intermediate nodes; both embed in [min-cost flow](/cs-glossary/minimum-cost-flow/).
