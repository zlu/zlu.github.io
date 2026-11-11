---
title: Formal language
synonyms:
  - Language
related:
  - Alphabet
  - Regular language
  - Automaton
  - Regular expression
  - Accepting state
---
A **formal language** $L$ over an [alphabet](/cs-glossary/alphabet/) $\Sigma$ is any set of strings from $\Sigma^*$. Not “English.” A collection of inputs you count as yes.

Examples: all valid emails, all digit strings matching `^[0-9]+$`, all event sequences that leave a door safe. One particular string is a *member* of $L$; $L$ itself is the set.

Grammars generate members; [automata](/cs-glossary/automaton/) accept or reject them. [Regular languages](/cs-glossary/regular-language/) are the ones finite automata / [regex](/cs-glossary/regular-expression/) can describe. Larger classes need pushdowns, Turing machines, and so on.
