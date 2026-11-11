---
title: Alphabet
synonyms:
  - Σ
related:
  - Formal language
  - Regular expression
  - Automaton
  - DFA
---
An **alphabet** $\Sigma$ is a finite nonempty set of symbols. Strings are finite sequences drawn from $\Sigma$; $\Sigma^*$ is every finite string including the empty one.

Coder view: the charset or the set of event names your machine understands. For a binary DFA, $\Sigma = \{0,1\}$. For a UI controller, $\Sigma$ might be `{press, release, tick}`.

Every [formal language](/cs-glossary/formal-language/) and every [automaton](/cs-glossary/automaton/) is relative to some $\Sigma$. Change the alphabet and you change which strings even exist, before acceptance comes into play.
