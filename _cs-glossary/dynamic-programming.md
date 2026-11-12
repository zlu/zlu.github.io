---
title: Dynamic programming
synonyms:
  - DP
  - Memoization
related:
  - Recursion
  - Divide and conquer
  - Greedy algorithm
  - DAG
  - Bellman-Ford
  - Floyd-Warshall
---
**Dynamic programming** solves problems with overlapping subproblems and optimal substructure by storing each subproblem’s answer once. Top-down form is [recursion](/cs-glossary/recursion/) with memoization; bottom-up form fills a table in dependency order (often a [DAG](/cs-glossary/dag/) of states).

State definition is the hard part: what is enough to continue optimally? Classic patterns: knapsack, LCS, edit distance, grid paths, interval DP, tree DP, [Floyd-Warshall](/cs-glossary/floyd-warshall/), [Bellman-Ford](/cs-glossary/bellman-ford/).

Versus [divide and conquer](/cs-glossary/divide-and-conquer/): DP reuses overlapping answers. Versus [greedy](/cs-glossary/greedy-algorithm/): DP considers multiple candidates per state when a single local choice is not enough.
