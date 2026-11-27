---
title: P-value
synonyms:
  - p value
  - Observed significance level
  - p(X)
keywords: p-value, significance, hypothesis testing, Type I error, statistics glossary, probability
related:
  - Hypothesis testing
  - Null hypothesis
  - Test statistic
  - Significance level
  - Type I error
  - T-test
  - Confidence interval
---
A **$p$-value** $p(X)$ is the probability, under the [null](/cs-glossary/null-hypothesis/), that the [test statistic](/cs-glossary/test-statistic/) is at least as extreme as the value observed. Small $p$ (e.g. $<\alpha=0.05$) ⇒ data unusual under $H_0$ ⇒ evidence against $H_0$.

Formally, for an observed $t_{\mathrm{obs}}$ from a test that rejects for large $|T|$,

$$
p=P\bigl(|T|\ge|t_{\mathrm{obs}}|\bigm| H_0\bigr).
$$

Reject $H_0$ at level $\alpha$ iff $p\le\alpha$ (equivalent to $t_{\mathrm{obs}}$ falling in the [critical region](/cs-glossary/critical-region/)).

**Not** the probability that $H_0$ is true, nor $P(\mathrm{Type\ I\ error}\mid\mathrm{data})$. Match one- vs two-sided definitions to the [$t$/$z$/$F$](/cs-glossary/t-test/) procedure you use. In R `lm` output, `Pr(>|t|)` is the two-sided $p$ for $H_0:\beta_j=0$.
