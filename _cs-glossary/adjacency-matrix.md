---
title: Adjacency matrix
related:
  - Graph
  - Adjacency list
  - Floyd-Warshall
---
An **adjacency matrix** is a $V \times V$ table where entry $(i,j)$ records whether edge $i \to j$ exists (or the weight). Space $\Theta(V^2)$ regardless of how sparse the [graph](/cs-glossary/graph/) is.

Edge query and update are $O(1)$. Iterating all edges is $\Theta(V^2)$. Natural fit for dense graphs and for [Floyd-Warshall](/cs-glossary/floyd-warshall/).

For sparse graphs, prefer an [adjacency list](/cs-glossary/adjacency-list/) so algorithms stay $O(V + E)$ instead of $O(V^2)$.
