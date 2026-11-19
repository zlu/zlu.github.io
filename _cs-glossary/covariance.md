---
title: Covariance
synonyms:
  - Cov
keywords: covariance, Cov(X,Y), uncorrelated, OLS, statistics
related:
  - Correlation
  - Variance
  - Independence
  - Ordinary least squares
---
**Covariance** $\mathrm{Cov}(X,Y)=E[XY]-\mu_X\mu_Y$ measures joint linear association. $\mathrm{Cov}(X,X)=\mathrm{Var}(X)$. Scaling: $\mathrm{Cov}(aX,bY)=ab\,\mathrm{Cov}(X,Y)$.

[Independence](/cs-glossary/independence/) $\Rightarrow$ $\mathrm{Cov}=0$; zero covariance (uncorrelated) does not imply independence. In simple [OLS](/cs-glossary/ordinary-least-squares/), $\hat\beta=\mathrm{Cov}(x,y)/\mathrm{Var}(x)$ (sample versions), and $\mathrm{Cov}(X,\varepsilon)=0$ is required for an [unbiased](/cs-glossary/unbiased-estimator/) slope.
