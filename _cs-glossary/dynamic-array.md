---
title: Dynamic array
synonyms:
  - Resizable array
  - Array list
  - Vector
related:
  - Array
  - Amortized analysis
  - Linked list
---
A **dynamic array** is a contiguous [array](/cs-glossary/array/) that grows (usually by doubling) when capacity is exceeded. Append is [amortized](/cs-glossary/amortized-analysis/) $O(1)$; occasional resize is $O(n)$.

This is `ArrayList`, `std::vector`, and Python’s `list`. Random access stays $O(1)$. Insert/delete in the middle stay $O(n)$.

Prefer over [linked lists](/cs-glossary/linked-list/) for most “list” workloads unless you need $O(1)$ splice at a known node without shifting.
