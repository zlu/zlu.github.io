---
title: DAG
synonyms:
  - Directed acyclic graph
related:
  - Graph
  - Topological sort
  - Dynamic programming
  - DFS
---
A **DAG** is a directed [graph](/cs-glossary/graph/) with no directed cycles. That restriction unlocks [topological sort](/cs-glossary/topological-sort/), unambiguous dependency order, and many [DP](/cs-glossary/dynamic-programming/) recurrences that process nodes in topo order.

Build pipelines, package dependencies, and spreadsheet formula graphs as DAGs. If a cycle appears, the model is wrong or you need a different algorithm (SCC condensation, explicit cycle handling).

Any directed graph’s strongly connected components contract to a DAG. That is why SCC algorithms show up next to topo sort in textbooks.
