---
title: Correlation
synonyms:
  - Pearson correlation
  - Correlation coefficient
keywords: correlation, Pearson, Cor(X,Y), association, statistics
related:
  - Covariance
  - Independence
  - Ordinary least squares
  - R-squared
---
**Correlation** (Pearson) standardises [covariance](/cs-glossary/covariance/):

$$
\mathrm{Cor}(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}\in[-1,1].
$$

$\pm 1$ means perfect linear relationship; $0$ means no *linear* association (nonlinear dependence can remain). Sample correlation uses sample SD and covariance. High correlation is not causation — regression [inference](/cs-glossary/statistical-inference/) still needs a model and assumptions.
