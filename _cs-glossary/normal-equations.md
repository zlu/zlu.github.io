---
title: Normal equations
synonyms:
  - Least-squares normal equations
  - OLS normal equations
keywords: normal equations, least squares, FOC, OLS, Sxx, Sxy, statistics glossary
related:
  - Ordinary least squares
  - Residual sum of squares
  - Linear regression
  - Fitted value
---
The **normal equations** are the first-order conditions for minimising [RSS](/cs-glossary/residual-sum-of-squares/) in [OLS](/cs-glossary/ordinary-least-squares/). For $y_i=\alpha+\beta x_i+\varepsilon_i$,

$$
\frac{\partial S_e^2}{\partial\alpha}=0,\qquad
\frac{\partial S_e^2}{\partial\beta}=0
$$

rearrange to

$$
n\hat\alpha+\hat\beta\sum x_i=\sum y_i,\qquad
\hat\alpha\sum x_i+\hat\beta\sum x_i^2=\sum x_i y_i.
$$

Solving yields $\hat\alpha=\bar y-\hat\beta\bar x$ and

$$
\hat\beta=\frac{S_{xy}}{S_{xx}},\quad
S_{xy}=\sum(x_i-\bar x)(y_i-\bar y),\quad
S_{xx}=\sum(x_i-\bar x)^2.
$$

Equivalently $\hat\beta=\mathrm{Cov}(x,y)/\mathrm{Var}(x)$. The Hessian of RSS is positive definite, so the stationary point is a minimum.
