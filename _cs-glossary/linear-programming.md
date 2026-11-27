---
title: Linear programming
synonyms:
  - LP
  - Linear optimisation
  - Linear optimization
keywords: linear programming, LP, simplex, polyhedron, optimisation, OR
related:
  - Integer programming
  - Simplex method
  - Objective function
  - Constraint
  - Feasible region
  - Mathematical programming
---
**Linear programming** (LP) optimises a linear [objective](/cs-glossary/objective-function/) over linear [constraints](/cs-glossary/constraint/):

$$
\min\{c^\top x:\ Ax\ge b,\ x\ge 0\}.
$$

Solved by [simplex](/cs-glossary/simplex-method/), interior-point, or network algorithms when structure allows. Relaxing integrality of an IP yields an LP relaxation.
