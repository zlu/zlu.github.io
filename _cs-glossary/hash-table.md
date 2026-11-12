---
title: Hash table
synonyms:
  - Hash map
  - Dictionary
  - Hash set
related:
  - Searching
  - Amortized analysis
  - Array
  - Binary search tree
---
A **hash table** maps keys to values (or stores a set of keys) by hashing into buckets of an [array](/cs-glossary/array/). Expected search, insert, and delete are $O(1)$ with a good hash and load factor; worst case is $O(n)$ if everything collides.

Collision strategies: chaining (lists/trees per bucket) or open addressing (probing). Rehashing when load grows is an [amortized](/cs-glossary/amortized-analysis/) cost.

Prefer for equality membership and associative maps. Prefer a sorted array / [BST](/cs-glossary/binary-search-tree/) when you need ordered iteration or range queries.
