---
title: Simplex method
synonyms:
  - Simplex algorithm
  - Dantzig simplex
keywords: simplex method, simplex algorithm, LP solver, pivot, optimisation
related:
  - Linear programming
  - Feasible region
  - Objective function
---
The **simplex method** solves [LPs](/cs-glossary/linear-programming/) by walking from vertex to adjacent vertex of the [feasible region](/cs-glossary/feasible-region/), improving the objective until no improving edge remains.

Each step pivots a basis of tight constraints. Practical workhorse for decades; worst-case exponential but fast on average. Interior-point methods are the main alternative for huge dense LPs.
