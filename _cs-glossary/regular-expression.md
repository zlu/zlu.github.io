---
title: Regular expression
synonyms:
  - Regex
  - Regexp
related:
  - Regular language
  - Kleene star
  - Alphabet
  - Thompson construction
  - ε-NFA
  - Formal language
---
A **regular expression** is a compact notation for a [regular language](/cs-glossary/regular-language/) over an [alphabet](/cs-glossary/alphabet/) $\Sigma$: symbols, concatenation, union (`|` / $+$), and [Kleene star](/cs-glossary/kleene-star/) (`*`), plus parentheses.

In theory, $L(r)$ is the language denoted by regex $r$. In code, engines add sugar (character classes, `+`, `?`, backreferences). Pure regular expressions (no backrefs) match exactly what finite automata can accept.

The constructive bridge is [Thompson construction](/cs-glossary/thompson-construction/): regex → [ε-NFA](/cs-glossary/epsilon-nfa/) → NFA → optional DFA. That is why “regex = finite state machine” is not a metaphor; it is an algorithm.
