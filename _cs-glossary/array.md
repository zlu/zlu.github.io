---
title: Array
synonyms:
  - Static array
  - Contiguous array
related:
  - Dynamic array
  - Linked list
  - Binary search
  - Sorting
---
An **array** stores elements in contiguous memory indexed by integers $0..n-1$ (or $1..n$). Random access is $O(1)$; inserting or deleting in the middle is $O(n)$ because other elements must shift.

Sorted arrays support [binary search](/cs-glossary/binary-search/) and are the usual input to textbook [sorting](/cs-glossary/sorting/) algorithms. Cache-friendly scans beat pointer-heavy structures on dense numeric data.

Fixed-capacity arrays do not grow; see [dynamic array](/cs-glossary/dynamic-array/) for amortized growth (`ArrayList`, `std::vector`, Python `list`).
