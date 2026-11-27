---
title: Spatial weights matrix
synonyms:
  - W matrix
  - Spatial weight matrix
keywords: spatial weights matrix, W matrix, contiguity, distance decay, Moran's I
related:
  - Spatial autocorrelation
  - Moran's I
  - Spatial lag model
  - Spatial error model
---
A **spatial weights matrix** $W$ is $n\times n$ with $w_{ij}$ measuring influence of location $j$ on $i$ — contiguity ($1$ for neighbours), inverse distance, exponential decay $e^{-d_{ij}/\mu}$, $k$-NN, or network distance.

Often **row-standardised** so each row sums to 1. Enters [Moran's I](/cs-glossary/morans-i/) and spatial regressions $\rho W y$ or $\lambda W u$.
