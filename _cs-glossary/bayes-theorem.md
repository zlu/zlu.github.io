---
title: Bayes' theorem
synonyms:
  - Bayes theorem
  - Bayes' rule
  - Bayesian update
keywords: Bayes theorem, Bayesian, prior likelihood posterior, diagnostic test, statistics
related:
  - Conditional probability
  - Prior
  - Likelihood
  - Posterior
  - Law of total probability
---
**Bayes' theorem** reverses [conditional probability](/cs-glossary/conditional-probability/):

$$
P(E\mid F)=\frac{P(F\mid E)P(E)}{P(F)}.
$$

In words: [posterior](/cs-glossary/posterior/) $\propto$ [likelihood](/cs-glossary/likelihood/) $\times$ [prior](/cs-glossary/prior/). Compute $P(F)$ with the [law of total probability](/cs-glossary/law-of-total-probability/).

Diagnostic template: prevalence $P(D)$, sensitivity $P(T\mid D)$, specificity $P(T'\mid D')$; target $P(D\mid T)$. Do not confuse $P(T\mid D)$ with $P(D\mid T)$.
