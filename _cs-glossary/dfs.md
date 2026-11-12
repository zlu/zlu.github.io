---
title: DFS
synonyms:
  - Depth-first search
  - Depth first search
related:
  - BFS
  - Graph
  - Stack
  - Recursion
  - Topological sort
  - Backtracking
  - Searching
---
**DFS** (depth-first search) explores a [graph](/cs-glossary/graph/) by going as deep as possible along each branch before backtracking. Implement with an explicit [stack](/cs-glossary/stack/) or with [recursion](/cs-glossary/recursion/) (the call stack).

Time $O(V + E)$ with adjacency lists. Discovery and finish times support cycle detection, [topological sort](/cs-glossary/topological-sort/) on DAGs, and strongly connected components (Kosaraju / Tarjan).

Prefer DFS when you care about path existence, structure (cycles, SCC), or [backtracking](/cs-glossary/backtracking/) search. Prefer [BFS](/cs-glossary/bfs/) when you need fewest edges from a source.
