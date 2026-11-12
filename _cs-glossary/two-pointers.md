---
title: Two pointers
synonyms:
  - Two-pointer technique
  - Twin pointers
related:
  - Binary search
  - Array
  - Searching
  - Sorting
---
The **two pointers** technique walks an [array](/cs-glossary/array/) (or string) with two indices that only move forward (or toward each other), turning a nested $O(n^2)$ scan into $O(n)$ when the structure allows it.

Patterns: opposite ends on a sorted array (pair sum), fast/slow (cycle detection, middle node), and sliding windows (subarray constraints). Often pairs with a prior [sort](/cs-glossary/sorting/).

Same complexity class as a careful single pass; the point is the invariant that lets you advance a pointer permanently without backtracking.
