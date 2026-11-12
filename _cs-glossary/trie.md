---
title: Trie
synonyms:
  - Prefix tree
  - Digital tree
related:
  - Hash table
  - Binary search tree
  - Searching
  - Radix sort
---
A **trie** stores strings (or digit sequences) in a tree keyed by successive characters. Shared prefixes share a path; end-of-word markers distinguish keys like `app` from `apple`.

Lookup, insert, and prefix query are $O(L)$ in the string length $L$, independent of how many keys are stored (aside from alphabet branching). Autocomplete and IP routing tables use the same idea.

Versus a [hash table](/cs-glossary/hash-table/): tries expose prefix structure; hashes do not. Memory can be large unless compressed (radix tree, Patricia).
