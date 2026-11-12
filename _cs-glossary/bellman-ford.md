---
title: Bellman-Ford
synonyms:
  - Bellman Ford algorithm
related:
  - Shortest path
  - Dijkstra
  - Graph
  - Dynamic programming
---
**Bellman-Ford** computes single-source [shortest paths](/cs-glossary/shortest-path/) even when some edge weights are negative (no negative cycle reachable from the source). It relaxes every edge $|V|-1$ times; an extra pass that still improves a distance detects a negative cycle.

[Time](/cs-glossary/time-complexity/) $O(VE)$. Slower than [Dijkstra](/cs-glossary/dijkstra/) on nonnegative instances, but the right tool when negatives appear or you need a negative-cycle check.

The recurrence “best path with at most $k$ edges” is a simple [dynamic programming](/cs-glossary/dynamic-programming/) view of the same algorithm.
