---
title: "Iterative local search (ILS)"
description: "Iterative local search escapes local optima with perturbation and acceptance criteria, with an AlgoScope demo that shows restarts after local search stalls."
keywords: "iterative local search, ILS, tabu search, metaheuristic, local optima, optimization visualization"
demo: "https://algoscape.zlu.me/optimization"
order: 30
related_posts:
  - "_posts/2025-05-12-iterative-local-search.md"
  - "_posts/2025-05-12-iterative-local-search-cn.md"
---

Iterative local search (ILS) improves a solution by alternating intense local search with controlled perturbation. After a local optimum is found, the incumbent is shaken just enough to leave its basin, then local search runs again. An acceptance rule decides whether the new local optimum replaces the incumbent — a simple structure that competes well on many combinatorial problems.

## Related ideas in the demo

The AlgoScape optimization view also includes tabu search, which forbids recently visited moves so the search does not cycle. Watching ILS and tabu side by side makes the tradeoff concrete: how hard you perturb, how long you forbid moves, and when you accept a worse candidate.

## Try it

Explore the [optimization algorithms demo](https://algoscape.zlu.me/optimization) (ILS and tabu search on a searchable landscape).
