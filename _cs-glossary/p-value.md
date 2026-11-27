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
A **$p$-value** tells you how unusual your result would be if the [null hypothesis](/cs-glossary/null-hypothesis/) were true. It is the probability of getting a [test statistic](/cs-glossary/test-statistic/) at least as extreme as the one you observed, assuming the null hypothesis and the test’s assumptions hold. A small $p$-value gives evidence against the null hypothesis.

Suppose you flip a coin 10 times and get 9 heads. If the coin is fair and the flips are independent, the probability of getting **9 or 10 heads** is about **1.1%**. So if you are testing whether the coin favors heads, the $p$-value is about **0.011**. If you are testing for bias in either direction, **0 or 1 heads** are equally extreme outcomes. Counting both tails gives a $p$-value of about **0.021**.

The probability is conditional on the null hypothesis. In the one-sided example, 0.011 is the probability of getting 9 or more heads **if the coin is fair**. The $p$-value alone cannot tell you how likely it is that the coin is fair given the flips.

Formally, for an observed $t_{\mathrm{obs}}$ from a test that rejects for large $\lvert T\rvert$,

$$
p = P\bigl(\lvert T\rvert \ge \lvert t_{\mathrm{obs}}\rvert \mid H_0\bigr).
$$

If you choose a [significance level](/cs-glossary/significance-level/) $\alpha$ before running the test, reject $H_0$ when $p\le\alpha$. This is equivalent to the observed statistic falling in the [critical region](/cs-glossary/critical-region/).

In R `lm` output, `Pr(>|t|)` is the two-sided $p$-value for $H_0:\beta_j=0$.
