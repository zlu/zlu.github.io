---
title: Beam Search
related:
  - Decoding
  - Sampling
---
**Beam search** keeps the top-$B$ partial hypotheses at each step, expanding and pruning by cumulative score. It is common in translation and speech but can produce bland text for open-ended chat.

For creative LLM use, temperature sampling or nucleus sampling is usually preferred; beam search still helps when you need high-probability structured outputs.
