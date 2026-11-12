---
title: Binary search tree
synonyms:
  - BST
related:
  - Binary tree
  - Binary search
  - Tree traversal
  - Hash table
  - Searching
---
A **binary search tree** ([binary tree](/cs-glossary/binary-tree/)) stores keys so that every left subtree key is $<$ the node and every right subtree key is $>$. Inorder traversal lists keys sorted.

Search, insert, and delete follow the same compare-and-descend idea as [binary search](/cs-glossary/binary-search/). Average $O(\log n)$ on random trees; worst case $O(n)$ when skewed. Balanced variants (AVL, red-black) restore $O(\log n)$ worst case.

Prefer a [hash table](/cs-glossary/hash-table/) for pure equality maps; prefer a BST when you need predecessor, successor, or range queries.
