---
title: Shortest path
synonyms:
  - Shortest paths
  - Pathfinding
related:
  - BFS
  - Dijkstra
  - Bellman-Ford
  - Floyd-Warshall
  - Graph
---
A **shortest path** between two nodes minimizes the sum of edge weights (or the hop count when every weight is 1). Algorithms differ by graph assumptions:

- Unweighted: [BFS](/cs-glossary/bfs/)
- Nonnegative weights, single source: [Dijkstra](/cs-glossary/dijkstra/)
- Negative weights allowed: [Bellman-Ford](/cs-glossary/bellman-ford/)
- All pairs, dense: [Floyd-Warshall](/cs-glossary/floyd-warshall/)
- Heuristic search on large implicit graphs: [A*](/algos/astar/)

Always state the weight model on exams. “Shortest” without weights usually means fewest edges.
