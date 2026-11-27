---
title: Event partition
synonyms:
  - Partition of the sample space
  - Partition (probability)
keywords: partition, mutually exclusive, exhaustive, law of total probability, Bayes, statistics glossary, probability
related:
  - Disjoint events
  - Law of total probability
  - Bayes' theorem
  - Sample space
  - Event
---
An **event partition** of $F\subseteq\Omega$ (often of the whole [sample space](/cs-glossary/sample-space/) $\Omega$) is a collection $E_1,\ldots,E_k$ that is pairwise [disjoint](/cs-glossary/disjoint-events/) and covers $F$: $\bigcup_{i=1}^k E_i=F$.

Partitions underwrite the [law of total probability](/cs-glossary/law-of-total-probability/) and [Bayes](/cs-glossary/bayes-theorem/): condition on each piece $E_i$, then reassemble.
