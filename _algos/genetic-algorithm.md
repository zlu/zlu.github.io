---
title: "Genetic algorithm"
description: "Genetic algorithms from selection, crossover, and mutation to an animated fitness demo that evolves a binary string toward more 1s."
keywords: "genetic algorithm, GA, evolutionary algorithm, selection crossover mutation, metaheuristic visualization"
demo: "https://algoscape.zlu.me/ga"
order: 20
related_posts:
  - "_posts/2025-05-12-genetic-algorithm-from-darwin-to-holland.md"
  - "_posts/2025-05-12-genetic-algorithm-from-darwin-to-holland-cn.md"
---

Genetic algorithms (GAs) search large solution spaces by maintaining a population of candidates and iteratively applying selection, crossover, and mutation — operators inspired by natural evolution. John Holland formalized the approach in *Adaptation in Natural and Artificial Systems* (1975); today GAs show up in scheduling, design optimization, and other combinatorial problems where gradients are unavailable.

## What the demo shows

The AlgoScape visualization evolves a population of binary strings. Fitness is the count of `1` bits, so you can see fitter individuals reproduce, crossover mix genes, and mutation keep diversity from collapsing. It is a toy objective, but the loop matches production GAs: evaluate → select → recombine → mutate → repeat.

## Try it

Run the [genetic algorithm visualization](https://algoscape.zlu.me/ga) and watch fitness climb across generations.
