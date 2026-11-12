---
title: Floyd-Warshall
synonyms:
  - Floyd Warshall algorithm
  - All-pairs shortest paths
related:
  - Shortest path
  - Dijkstra
  - Dynamic programming
  - Graph
---
**Floyd-Warshall** computes all-pairs [shortest paths](/cs-glossary/shortest-path/) on a weighted [graph](/cs-glossary/graph/) (positive or negative edges, no negative cycles) via [dynamic programming](/cs-glossary/dynamic-programming/): allow paths that may use intermediate nodes $\{1,\ldots,k\}$ and update a dense distance matrix.

[Time](/cs-glossary/time-complexity/) $\Theta(V^3)$, space $\Theta(V^2)$. Simple to code; fine for dense graphs with hundreds of nodes. For sparse graphs and few sources, repeated [Dijkstra](/cs-glossary/dijkstra/) (nonnegative) or [Bellman-Ford](/cs-glossary/bellman-ford/) is usually better.
