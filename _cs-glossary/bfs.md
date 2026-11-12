---
title: BFS
synonyms:
  - Breadth-first search
  - Breadth first search
related:
  - DFS
  - Graph
  - Queue
  - Shortest path
  - Dijkstra
  - Searching
---
**BFS** (breadth-first search) explores a [graph](/cs-glossary/graph/) level by level from a source using a [queue](/cs-glossary/queue/): dequeue a node, enqueue its unseen neighbors. Nodes are typically marked visited when enqueued so each is processed once.

On an unweighted graph, the first time BFS reaches a node is a minimum-edge [shortest path](/cs-glossary/shortest-path/) from the source. Time $O(V + E)$ with adjacency lists; space $O(V)$ for the queue and visited set.

Use BFS for shortest paths in hops, bipartite checks, and level-order tree traversal. For weighted edges use [Dijkstra](/cs-glossary/dijkstra/) or another weighted algorithm, not plain BFS.
