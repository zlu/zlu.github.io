---
title: Z-score
synonyms:
  - Standard score
  - Standardised value
keywords: z-score, standard score, standard normal, CLT, statistics
related:
  - Normal distribution
  - Central limit theorem
  - T-test
  - Standard deviation
---
A **$z$-score** (standard score) rescales a value as distance from the mean in SD units: $z=(x-\mu)/\sigma$. For $X\sim N(\mu,\sigma^2)$, $Z=(X-\mu)/\sigma\sim N(0,1)$.

The [CLT](/cs-glossary/central-limit-theorem/) uses a $z$-scale for $\bar X$: $\sqrt{n}(\bar X-\mu)/\sigma$. When $\sigma$ is replaced by $S$, large-sample $z$ procedures become [$t$](/cs-glossary/t-test/) procedures in exact Normal models.
