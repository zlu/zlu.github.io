---
title: Adjacency list
related:
  - Graph
  - Adjacency matrix
  - BFS
  - DFS
---
An **adjacency list** stores, for each vertex, a list (or vector) of neighbors. For a [graph](/cs-glossary/graph/) with $V$ vertices and $E$ edges, space is $\Theta(V + E)$.

Best default for sparse graphs. Iterating all edges is $O(V + E)$; checking a specific edge $(u,v)$ is $O(\deg(u))$ unless you add a set/hash per row.

[BFS](/cs-glossary/bfs/) and [DFS](/cs-glossary/dfs/) textbooks almost always assume adjacency lists. Dense graphs with many edge queries may prefer an [adjacency matrix](/cs-glossary/adjacency-matrix/).
