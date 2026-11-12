---
title: Backtracking
synonyms:
  - Generate and test
related:
  - DFS
  - Recursion
  - Searching
  - Dynamic programming
---
**Backtracking** searches a tree of partial candidates with [DFS](/cs-glossary/dfs/): extend a partial solution, recurse, then undo the extension (the “backtrack”) to try the next option. Prune when a partial answer cannot lead to a valid complete one.

N-queens, Sudoku, subset-sum search, and constraint solvers use this pattern. Time is often exponential; pruning and ordering heuristics matter more than micro-optimizing the recursion.

When overlapping subproblems appear with a clear value to store, lift the search into [DP](/cs-glossary/dynamic-programming/). When you only need any feasible solution and the graph is explicit, plain DFS/BFS may be enough.
