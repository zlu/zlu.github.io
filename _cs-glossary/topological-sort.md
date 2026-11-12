---
title: Topological sort
synonyms:
  - Toposort
  - Topological ordering
related:
  - DFS
  - Graph
  - DAG
  - BFS
---
A **topological sort** of a directed acyclic [graph](/cs-glossary/graph/) ([DAG](/cs-glossary/dag/)) is a linear order of vertices such that every edge $u \to v$ has $u$ before $v$. Build schedules, compile dependency order, and course-prerequisite lists this way.

Two standard algorithms: [DFS](/cs-glossary/dfs/) finish-time order (reverse postorder), and Kahn’s algorithm (repeatedly dequeue indegree-0 nodes with a [queue](/cs-glossary/queue/), i.e. a [BFS](/cs-glossary/bfs/)-style peel). A cycle means no topological order exists; both algorithms can detect that.

Time $O(V + E)$. If the graph might have cycles, treat “no topo order” as a first-class outcome, not an afterthought.
