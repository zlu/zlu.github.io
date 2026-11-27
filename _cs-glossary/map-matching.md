---
title: Map matching
synonyms:
  - GPS map matching
  - Snap to road
keywords: map matching, GPS, GIS, digital road network, transport telematics
related:
  - GIS
  - Vector data model
  - Shortest path
  - Graph
---
**Map matching** integrates noisy GPS (and dead-reckoning) traces with a digital road [network](/cs-glossary/vector-data-model/) to infer the correct link and position on that link.

Hard in dense urban graphs with many candidate segments. Weighting scores combine proximity and heading; underpins sat-nav, bus arrival, and crash mapping pipelines.
