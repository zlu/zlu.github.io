---
title: Moran's I
synonyms:
  - Moran I
  - Moran’s I statistic
keywords: Moran's I, spatial autocorrelation test, spatial statistics, statistics glossary
related:
  - Spatial autocorrelation
  - Spatial weights matrix
  - Spatial lag model
  - Hypothesis testing
---
**Moran's I** tests [spatial autocorrelation](/cs-glossary/spatial-autocorrelation/) given weights $W$:

$$
I=\frac{n}{\sum_i\sum_j w_{ij}}\frac{\sum_i\sum_j w_{ij}(y_i-\bar y)(y_j-\bar y)}{\sum_i(y_i-\bar y)^2}.
$$

$I\approx 0$: little spatial structure; near $+1$: clustering of similar values; near $-1$: checkerboard dissimilarity. $H_0$: no spatial autocorrelation. First checkpoint before choosing [SLM](/cs-glossary/spatial-lag-model/) vs [SEM](/cs-glossary/spatial-error-model/).
