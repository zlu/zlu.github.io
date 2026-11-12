---
title: Recursion
synonyms:
  - Recursive algorithm
related:
  - Stack
  - Divide and conquer
  - Dynamic programming
  - DFS
  - Backtracking
---
**Recursion** solves a problem by calling the same procedure on a smaller instance until a base case. The call [stack](/cs-glossary/stack/) holds return addresses and local frames; deep recursion can overflow even when the math is fine.

It is the natural expression of [divide and conquer](/cs-glossary/divide-and-conquer/), [DFS](/cs-glossary/dfs/) on trees/graphs, and [backtracking](/cs-glossary/backtracking/). Tail-recursive forms can be rewritten as loops; languages differ on whether they optimize that.

Always state the base case and the measure that decreases. Infinite recursion is usually a missing base case or a measure that does not shrink.
