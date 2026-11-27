---
title: Assignment problem
synonyms:
  - Bipartite matching with costs
  - Linear assignment
keywords: assignment problem, Hungarian algorithm, bipartite matching, OR
related:
  - Hungarian algorithm
  - Integer programming
  - Binary variable
  - Minimum cost flow
---
The **assignment problem** pairs $n$ agents to $n$ tasks at minimum total cost: binary $x_{ij}\in\{0,1\}$ with row/column sum constraints (each agent/task used once).

Solved by the [Hungarian algorithm](/cs-glossary/hungarian-algorithm/) or as a [min-cost flow](/cs-glossary/minimum-cost-flow/)/bipartite matching IP. Classic crew/vehicle–task allocation template.
