---
title: Union-find
synonyms:
  - Disjoint-set
  - DSU
  - Merge-find set
related:
  - Graph
  - Greedy algorithm
  - Amortized analysis
---
**Union-find** (disjoint-set union, DSU) maintains a partition of elements under two operations: `find` (which set is this in?) and `union` (merge two sets). Path compression plus union-by-rank gives nearly $O(1)$ [amortized](/cs-glossary/amortized-analysis/) time per op (inverse Ackermann).

Kruskal’s MST, connected-component maintenance, and “are these two nodes already linked?” queries are the standard uses. It is a [graph](/cs-glossary/graph/) connectivity tool without building full adjacency structures for every query mix.
