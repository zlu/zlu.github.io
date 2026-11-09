---
layout: post
title: "Automata Basics for Coders, Part III: Regular Expressions and ε-NFAs"
date: 2026-11-09
comments: true
tags:
  - automata
  - formal languages
  - theory of computation
  - regex
  - NFA
description: "Linz Chapter 3 in plain terms: regular expressions, Thompson-style ε-NFA building blocks, membership by splitting, and lazy removal of ε-moves."
---

This continues from [Part II](/2026/11/07/automata-basics-for-coders-part-2/) (DFAs, NFAs, subset construction). The text is Linz and Rodger, [*An Introduction to Formal Languages and Automata*](https://www.betterreads.dev/discover/ia_an-introduction-to-formal-languages-and-automata-6th-edition?q=An%20Introduction%20to%20Formal%20Languages%20and%20Automata){:target="_blank" rel="noopener noreferrer"}, **Chapter 3**. Linz writes **λ** for the empty string and for empty moves; many courses write **ε**. Same objects.

Content:

1. Regular expressions and the languages they denote
2. Membership by parsing a regex
3. Thompson construction: regex → ε-NFA
4. ε-closure and lazy ε-NFA → NFA
5. Cheat sheet and self-check

Pipeline:

```text
regex  →  ε-NFA (building blocks)  →  NFA (remove ε)  →  (optional) DFA
```

Part II already covered NFA → DFA. This post stops after the NFA without ε-edges.

---

## Regular expressions

### What a regular expression is

A **regular expression** (regex) is a compact notation for a set of strings over an alphabet $\Sigma$. The set it names is its **language** $L(r)$. In code you already write the same idea (`/0(0|1)*1+/`, `grep`, validators): the pattern is the expression; the strings that match are the language.

Why bring in automata? Regular expressions and the finite accepters from Part II (DFAs and NFAs) describe exactly the same class of languages: the **regular languages**. This chapter makes one direction constructive. Build an ε-NFA from the regex (Thompson), remove ε-moves to get an ordinary NFA, then optionally subset-construct a DFA. That is the pipeline at the top of this post.

### Syntax and meaning

Fix an alphabet $\Sigma$. Regular expressions over $\Sigma$ are built inductively. Each expression $r$ denotes a language $L(r) \subseteq \Sigma^*$.

| Expression | Name | Language |
|------------|------|----------|
| $\emptyset$ | empty set | $\emptyset$ |
| $\lambda$ | empty string | $\{\lambda\}$ |
| $a$ ($a \in \Sigma$) | symbol | $\{a\}$ |
| $(r+s)$ or $(r\|s)$ | union | $L(r) \cup L(s)$ |
| $(rs)$ | concatenation | $\{uv \mid u \in L(r),\ v \in L(s)\}$ |
| $r^*$ | [Kleene star](/cs-glossary/kleene-star/) | $\{\lambda\} \cup L(r) \cup L(r)L(r) \cup \cdots$ |
| $r^+$ | one or more | $L(r)\,L(r)^*$ (does **not** contain $\lambda$ unless $L(r)$ does) |
| $r?$ | zero or one | $L(r) \cup \{\lambda\}$ |

**Precedence** (tight → loose): `*`, `+`, `?` bind tighter than concatenation, which binds tighter than `|` / `+` (union). Parentheses override.

Example:

$a \vert bc^{\ast}$ means $\lbrace a \rbrace \cup \lbrace b \rbrace\lbrace c \rbrace^{\ast}$, not $(\lbrace a \rbrace \cup \lbrace b \rbrace)\lbrace c \rbrace^{\ast}$.

### Toy Example

Strings over $\lbrace 0,1 \rbrace$ that

1. start with $0$,  
2. then any string in $\lbrace 0,1 \rbrace^{\ast}$,  
3. end with one or more $1$s.

Build left to right: $0$, then $(0\|1)^*$, then $1^+$. Glue:

$$
r = 0(0|1)^*1^+
$$

| string | in $L(r)$? |
|--------|------------|
| $01$ | yes |
| $00111$ | yes |
| $1$ | no (must start with $0$) |
| $0$ | no (need $\ge 1$ final $1$) |
| $010$ | no (must end with $1$) |

---

## Membership by splitting

Given a regex, decide whether a concrete string is in its language by parsing the operators, then covering the string with pieces that match the subexpressions.

**Toy regex:** $(0^+\|1^*0)^+$

- $0^+$ = one or more $0$s  
- $1^*0$ = zero or more $1$s, then one $0$  
- $(0^+\|1^*0)$ = one **block** of either form  
- outer $^+$ = one or more blocks concatenated  

| string | split | in $L$? |
|--------|-------|---------|
| $0$ | $(0)$ | yes |
| $10$ | $(10)$ | yes |
| $00$ | $(00)$ or $(0)(0)$ | yes |
| $010$ | $(0)(10)$ | yes |
| $01$ | leftover $1$ after a $0$-block, or $01$ is not a block | no |
| $1$ | $1$ alone is not a block | no |

Method: walk left to right; if the whole string can be partitioned into legal blocks, accept; otherwise reject.

---

## Thompson construction (regex → ε-NFA)

### Theorem (Linz Ch 3.2)

Every regular expression denotes a regular language, and there is an algorithm that builds an NFA (with $\lambda$/ε-moves) accepting exactly that language.

You do not need Thompson to know what a regex means, and for a tiny pattern you would just draw the NFA (or DFA) by hand with letter edges only. No ε required.

**Thompson construction** is one mechanical recipe that scales to an arbitrary regex tree: walk the operators and emit an ε-NFA. It is not the smallest machine, and ε is not a deep feature of regular languages. ε is glue this particular recipe invents so every piece has one entrance and one exit and the operators always wire the same way. That makes the translation an algorithm and the inductive proof easy ("every regex has an NFA"). The language is the same; the graph is just fatter. Then you delete the glue.

1. **Base case.** Each letter in the regex becomes a two-state machine with one letter-edge (diagram below).  
2. **Operators.** Concatenation, `|`, `*`, and unary `+` do **not** get letter-edges. You only add ε-edges that connect machines you already built. The four recipes are below.  
3. **One entrance, one exit** on every finished piece, so you can plug pieces into larger recipes.

ε means: move without reading an input symbol.

### Building blocks

Think of each finished piece as a black box with one entrance and one exit. The letter cases are the only boxes that actually read input. Every other case is: take finished boxes, add a few ε-edges (and maybe one scaffolding start / one scaffolding accept).

**Symbol $a$.** One edge labeled $a$. This is the only place a letter appears on an edge.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> s
    s --> f: a
    f --> [*]
</div>

**Concatenation (regex $RS$).** You already have boxes $R$ and $S$. Connect them in order: $R \xrightarrow{\varepsilon} S$. Start at $R$'s entrance; accept at $S$'s exit.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> R
    R --> S: ε
    S --> [*]
</div>

**Union (regex $R\|S$).** You already have boxes $R$ and $S$. The recipe refuses to give the machine two starts, so it adds a scaffolding start $N$ with ε into both boxes, and a scaffolding accept $A$ that both boxes ε into. By hand you might put a $0$-edge and a $1$-edge out of one state and skip $N$ and $A$. Thompson does not optimize; it always forks and joins with ε.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> N
    N --> R: ε
    N --> S: ε
    R --> A: ε
    S --> A: ε
    A --> [*]
</div>

**Star (regex $R^*$).** You already have a box $R$. Add a fresh start $N$ and a fresh accept $A$.

- $N \xrightarrow{\varepsilon} R$ (do the body)  
- $R \xrightarrow{\varepsilon} A$ (stop)  
- $R \xrightarrow{\varepsilon} R$ (loop for another copy)  
- $N \xrightarrow{\varepsilon} A$ (bypass: zero copies)

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> N
    N --> R: ε
    N --> A: ε
    R --> R: ε
    R --> A: ε
    A --> [*]
</div>

**Plus (regex $R^+$).** Same diagram as star, but delete the bypass $N \xrightarrow{\varepsilon} A$. You must enter $R$ at least once.

The next section builds $(0\|1)0$ with these recipes in order: two symbol machines, then union, then concat.

### Example: $(0\|1)0$

Layers:

1. machines for $0$ and for $1$: $p \xrightarrow{0} q$ and $r \xrightarrow{1} s$  
2. union → $0\|1$: one start $n$, ε into both machines; both exits ε into join $j$  
3. concat symbol $0$: $j \xrightarrow{\varepsilon} t \xrightarrow{0} u$

So yes: $n$ and $j$ are scaffolding. $p$ and $r$ are the real starts of the two letter machines; $n$ is a bogus outer start that ε-spawns into both, because the recipe insists on one start per finished piece. Same for $j$ on the way out. The accepting behaviour lives on the letter-edges $p \xrightarrow{0} q$ and $r \xrightarrow{1} s$ (then the final $0$).

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> n
    n --> p: ε
    n --> r: ε
    p --> q: 0
    r --> s: 1
    q --> j: ε
    s --> j: ε
    j --> t: ε
    t --> u: 0
    u --> [*]
</div>

Accepting path for string $10$: $n \xrightarrow{\varepsilon} r \xrightarrow{1} s \xrightarrow{\varepsilon} j \xrightarrow{\varepsilon} t \xrightarrow{0} u$.

---

## Removing ε-moves (lazy evaluation)

### Why remove ε?

ε was invented for the construction. Thompson needed a way to wire operator scaffolding (fork for `|`, join, bypass for `*`) without pretending those wires read a letter. So an ε-edge is exactly: jump from one invented state to another with **no input**.

Once the regex has been turned into a graph, those free jumps are leftover glue. You do not want them in the machine you keep. Collapse every ε-path into the state names (via ε-closure below). What remains is an ordinary NFA: edges only on real alphabet symbols, same language. Then Part II's NFA → DFA applies as written.

### What "lazy" means here

Eager would mean: invent a new state for every subset of the old states (exponentially many). Most of those subsets never show up on any input.

**Lazy** means: only build subsets you actually reach. Start from the ε-closure of the ε-NFA start. From each subset you have, try each alphabet symbol, then take ε-closure again. Keep a new subset only if you have not seen it yet.

Same "reachable only" idea as subset construction in Part II. The difference is the ε-closure wrap on each letter step, so free moves disappear into the subset names.

### ε-closure

$\varepsilon\text{-closure}(S)$ = all states reachable from $S$ by **zero or more ε-edges** (including the states in $S$). That is the "chase free moves" step.

### Algorithm (ε-NFA → NFA without ε)

```text
NFA_start ← ε-closure({ ε-NFA start })
Worklist ← { NFA_start }

while Worklist not empty:
    take set S
    for each symbol σ in Σ:
        U ← { states reachable from some q ∈ S by a σ-edge }
        T ← ε-closure(U)
        record δ_NFA(S, σ) = T
        if T is new: add T to Worklist

Accepting: any set that contains an ε-NFA accept state
```

Each new state is a set. Each edge is on a letter. ε only appears inside the closure helper, not in the final table.

### Tiny example

ε-NFA for $(a\|b)$ (fragment of the example above, before the final $0$):

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> n
    n --> p: ε
    n --> r: ε
    p --> q: a
    r --> s: b
    q --> j: ε
    s --> j: ε
    j --> [*]
</div>

Steps:

1. $S_0 = \varepsilon\text{-closure}(\{n\}) = \{n,p,r\}$  
2. on $a$: move $\{q\}$, then $\varepsilon\text{-closure}(\{q\}) = \{q,j\} = S_1$ (accept)  
3. on $b$: move $\{s\}$, then $\varepsilon\text{-closure}(\{s\}) = \{s,j\} = S_2$ (accept)  
4. from $S_1$ / $S_2$ on $a,b$: often $\emptyset$ if no further letter-edges  

Final NFA table has columns only for $a,b$ (no ε column). State names are sets.

---

## Cheat sheet

```text
regex ops     | concat * + ?   (parenthesise when unsure)
a* vs a+      * includes λ; + does not
Thompson      letter edges from symbols; ε wires operators
ε-closure(S)  S plus everything reachable by ε only
lazy → NFA    start = ε-closure(start); on σ: move then close
membership    parse operators; cover the string with blocks
```

---

## Self-check

1. Write a regex for strings over $\{a,b\}$ that start with $a$ and end with $b$.  
2. Is $b$ in $L((a^+\|b^*a)^+)$?  
3. In Thompson's $R^*$, which ε-edge is deleted to obtain $R^+$?  
4. If $p \xrightarrow{\varepsilon} q \xrightarrow{\varepsilon} r$, what is $\varepsilon\text{-closure}(\{p\})$?  
5. Why does the final NFA table after lazy evaluation have no ε column?

### Answers

**1.** $a(a \vert b)^{\ast}b$ (also $ab$ if the middle may be empty and ends-with-$b$ is required; both need care for the single-symbol string $ab$). A standard form: $a(a \vert b)^{\ast}b$.

**2.** No. Blocks are $a^+$ or $b^*a$. A lone $b$ matches neither.

**3.** The bypass from the new start of the star straight to the new accept.

**4.** $\{p,q,r\}$.

**5.** Every ε-move was absorbed into ε-closure; remaining edges are on alphabet symbols only.

---

## References

- Linz and Rodger, 7th ed., Chapter 3 (and λ-transitions in Chapter 2.2).  
- [Part I](/2026/11/06/automata-basics-for-coders-part-1/), [Part II](/2026/11/07/automata-basics-for-coders-part-2/).
