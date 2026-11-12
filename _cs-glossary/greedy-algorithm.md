---
title: Greedy algorithm
synonyms:
  - Greedy
  - Greedy method
related:
  - Dijkstra
  - Huffman coding
  - Dynamic programming
  - Divide and conquer
---
A **greedy algorithm** builds a solution by always taking the locally best choice that looks safe under an exchange or matroid-style argument. No backtracking over earlier choices.

Examples: [Dijkstra](/cs-glossary/dijkstra/) (nonnegative shortest paths), interval scheduling by earliest finish, Prim/Kruskal for MSTs, [Huffman coding](/cs-glossary/huffman-coding/). Proofs usually show that some optimal solution agrees with the greedy choice, then induct.

Greedy is fast when it works and wrong when the local choice destroys global structure (many knapsack and path problems). If you cannot prove a greedy choice property, try [DP](/cs-glossary/dynamic-programming/) instead.
