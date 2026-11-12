---
title: Searching
synonyms:
  - Search
  - Search algorithm
related:
  - Linear search
  - Binary search
  - BFS
  - DFS
  - Hash table
  - Binary search tree
---
**Searching** finds a target key (or reports absence) in a collection. Cost depends on structure: unsorted array or list needs [linear search](/cs-glossary/linear-search/); sorted array admits [binary search](/cs-glossary/binary-search/); [hash tables](/cs-glossary/hash-table/) aim for expected $O(1)$; trees and graphs use walks such as [BST](/cs-glossary/binary-search-tree/) search, [BFS](/cs-glossary/bfs/), and [DFS](/cs-glossary/dfs/).

Coder view: pick the structure for the query mix you actually have. Sorting once to enable binary search pays off for many lookups. Hashing wins for equality membership. Graphs need an explicit traversal, not an array index.

Exams usually want the invariant and complexity, not the library call: loop bounds for binary search, visited-set for graph search, expected vs worst case for hashing.
