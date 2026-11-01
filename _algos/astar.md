---
title: "A* pathfinding"
description: "How A* finds shortest paths with f(n) = g(n) + h(n), compared with Dijkstra and greedy best-first search — plus a browser grid demo."
keywords: "A* algorithm, A-star pathfinding, heuristic search, Dijkstra, shortest path visualization"
demo: "https://algoscape.zlu.me/astar"
order: 10
related_posts:
  - "_posts/2025-05-09-visualizing-a-star-search.md"
  - "_posts/2025-05-09-可视化-A-Star-搜索.md"
---

A* is a heuristic search algorithm for finding a low-cost path from a start node to a goal. At each step it expands the open node with the smallest estimated total cost:

$$ f(n) = g(n) + h(n) $$

where $g(n)$ is the cost from the start to $n$, and $h(n)$ is a heuristic estimate of the remaining cost to the goal. With an admissible heuristic (one that never overestimates), A* is optimally efficient among algorithms that expand nodes in order of increasing $f$.

## Why visualize it

On a grid with obstacles, you can watch the frontier expand, see how the heuristic biases search toward the goal, and compare that behavior with uniform-cost search. The live demo lets you place walls and move start/goal so the open/closed sets update in real time.

## Try it

Open the [A* pathfinding demo on AlgoScape](https://algoscape.zlu.me/astar) and step through a search on your own map.
