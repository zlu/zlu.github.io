---
title: Dijkstra
synonyms:
  - Dijkstra's algorithm
  - Dijkstra shortest path
related:
  - Shortest path
  - BFS
  - Bellman-Ford
  - Priority queue
  - Graph
  - Greedy algorithm
---
**Dijkstra’s algorithm** computes single-source [shortest paths](/cs-glossary/shortest-path/) on a directed or undirected [graph](/cs-glossary/graph/) with nonnegative edge weights. It repeatedly settles the unsettled node with smallest tentative distance ([greedy](/cs-glossary/greedy-algorithm/) choice), then relaxes its outgoing edges.

With a binary [heap](/cs-glossary/heap/) / [priority queue](/cs-glossary/priority-queue/), time is $O((V + E) \log V)$ (or $O(E + V \log V)$ with a Fibonacci heap in theory). Nonnegative weights are required; negative edges need [Bellman-Ford](/cs-glossary/bellman-ford/).

When every edge weight is $1$, Dijkstra specializes to [BFS](/cs-glossary/bfs/). In code, the decrease-key vs “insert duplicates and skip stale entries” detail is the usual implementation fork.
