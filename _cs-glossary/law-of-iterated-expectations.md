---
title: Law of iterated expectations
synonyms:
  - Tower property
  - Law of total expectation
keywords: law of iterated expectations, tower property, conditional expectation, statistics glossary, probability
related:
  - Conditional expectation
  - Expectation
  - Law of total probability
  - Exogeneity
---
The **law of iterated expectations** (tower property): for random variables $X_1,X_2$,

$$
E\bigl[E[X_1\mid X_2]\bigr]=E[X_1].
$$

Taking $X_2=X$ and $X_1=\varepsilon$ with $E[\varepsilon\mid X]=0$ immediately gives $E[\varepsilon]=0$ — the bridge from conditional mean zero to unconditional mean zero in [regression](/cs-glossary/linear-regression/).
