---
title: Linear search
synonyms:
  - Sequential search
  - Brute-force search
related:
  - Searching
  - Binary search
  - Array
  - Linked list
---
**Linear search** scans elements one by one until it finds the target or exhausts the collection. Works on unsorted [arrays](/cs-glossary/array/) and [linked lists](/cs-glossary/linked-list/); no preprocessing beyond what you already have.

[Time](/cs-glossary/time-complexity/) $O(n)$ worst and average for a successful search in a random location; $O(1)$ best when the target is first. No extra memory.

Use when $n$ is small, data is unsorted and searched once, or you cannot reorder or index the structure. For many lookups on a static list, sort once and switch to [binary search](/cs-glossary/binary-search/), or put keys in a [hash table](/cs-glossary/hash-table/).
