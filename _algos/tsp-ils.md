---
title: "TSP with iterative local search"
description: "Traveling salesman tours improved by iterative local search — watch city routes shorten as perturbation and local improvement iterate."
keywords: "traveling salesman problem, TSP visualization, iterative local search, combinatorial optimization"
demo: "https://algoscape.zlu.me/tsp-ils"
order: 50
---

The traveling salesman problem (TSP) asks for a shortest tour that visits each city once and returns to the start. Exact solvers exist for moderate sizes; for larger instances, metaheuristics such as iterative local search build strong tours by improving a route locally, then perturbing and improving again.

## What you will see

The AlgoScape TSP view animates candidate tours as ILS runs. Pair it with the broader [ILS guide](/algos/iterative-local-search/) if you want the algorithmic framing; this page is the path-length intuition on a concrete map of cities.

## Try it

Open the [TSP + ILS demo](https://algoscape.zlu.me/tsp-ils) and watch tour length drop across iterations.
