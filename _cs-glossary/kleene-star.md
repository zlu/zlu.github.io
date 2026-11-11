---
title: Kleene star
synonyms:
  - Kleene closure
related:
  - Regular expression
  - Regular language
  - Alphabet
---
**Kleene star** (or Kleene closure) is the unary “zero or more” operation on languages (and on [regular expressions](/cs-glossary/regular-expression/)).

If $L$ is a [formal language](/cs-glossary/formal-language/), $L^*$ contains the empty string $\lambda$ (or $\varepsilon$) and every finite concatenation of strings from $L$. So $\{\mathtt{a}\}^*$ is $\{\lambda, \mathtt{a}, \mathtt{aa}, \mathtt{aaa}, \ldots\}$.

In a regex, `a*` matches that same set. Star is why [regular languages](/cs-glossary/regular-language/) are closed under unbounded repetition without leaving the finite-automaton world: Thompson builds a small [ε-NFA](/cs-glossary/epsilon-nfa/) loop for each `*`.
