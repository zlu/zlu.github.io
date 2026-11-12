---
title: Tree traversal
synonyms:
  - Tree walk
  - Inorder
  - Preorder
  - Postorder
  - Level order
related:
  - Binary tree
  - DFS
  - BFS
  - Binary search tree
---
**Tree traversal** visits every node of a tree in a defined order. Depth-first styles: preorder (node, left, right), inorder (left, node, right), postorder (left, right, node). Level-order is [BFS](/cs-glossary/bfs/) by depth.

Inorder on a [BST](/cs-glossary/binary-search-tree/) yields sorted keys. Expression trees use inorder/postorder for print and evaluation. All visit $n$ nodes in $\Theta(n)$ time for an $n$-node tree.

Recursive [DFS](/cs-glossary/dfs/) is the usual teaching form; iterative versions use an explicit [stack](/cs-glossary/stack/) or [queue](/cs-glossary/queue/).
