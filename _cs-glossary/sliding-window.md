---
title: Sliding window
synonyms:
  - Window technique
related:
  - Two pointers
  - Array
  - Deque
  - Searching
---
A **sliding window** maintains a contiguous subarray (or substring) `[left, right]` while `right` advances over the input and `left` catches up when a constraint breaks. Updates are $O(1)$ or $O(\log n)$ per move if you keep a running sum, count map, or [deque](/cs-glossary/deque/) of candidates.

Classic problems: longest substring with at most $k$ distinct chars, minimum window covering a target, fixed-length window aggregates. It is a structured form of [two pointers](/cs-glossary/two-pointers/).

Works when the validity predicate is monotone in the window size for a fixed right endpoint. If not monotone, window logic does not apply; use other search or DP.
