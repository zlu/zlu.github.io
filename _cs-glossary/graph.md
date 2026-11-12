---
title: Graph
synonyms:
  - Network
related:
  - DAG
  - BFS
  - DFS
  - Shortest path
  - Adjacency list
  - Adjacency matrix
---
A **graph** is a set of vertices plus a set of edges (undirected or directed, optionally weighted). It models social links, road maps, dependency relations, and state spaces.

Represent with an [adjacency list](/cs-glossary/adjacency-list/) (sparse, $O(V + E)$ space) or [adjacency matrix](/cs-glossary/adjacency-matrix/) (dense, $O(V^2)$). Core algorithms: [BFS](/cs-glossary/bfs/), [DFS](/cs-glossary/dfs/), [shortest paths](/cs-glossary/shortest-path/), MSTs, [topological sort](/cs-glossary/topological-sort/) on [DAGs](/cs-glossary/dag/).

Always state directed vs undirected and weighted vs unweighted before quoting complexities.
