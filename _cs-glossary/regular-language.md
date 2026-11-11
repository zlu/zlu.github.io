---
title: Regular language
related:
  - Regular expression
  - DFA
  - NFA
  - Formal language
  - Alphabet
  - Kleene star
---
A **regular language** is a [formal language](/cs-glossary/formal-language/) that some finite automaton accepts, or equivalently that some [regular expression](/cs-glossary/regular-expression/) denotes.

Same class, three faces: [DFA](/cs-glossary/dfa/), [NFA](/cs-glossary/nfa/) / [ε-NFA](/cs-glossary/epsilon-nfa/), and regex. Constructions convert among them. Closed under union, concatenation, [Kleene star](/cs-glossary/kleene-star/), complement, and intersection.

Not every language is regular. $\{a^n b^n \mid n \geq 0\}$ needs unbounded memory to match counts; finite state is not enough. Lexers and many validators stay inside the regular fragment on purpose.
