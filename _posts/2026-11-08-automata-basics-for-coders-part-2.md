---
layout: post
title: "Automata Basics for Coders, Part II: DFAs, NFAs, and Subset Construction"
date: 2026-11-08
comments: true
tags:
  - automata
  - formal languages
  - theory of computation
  - DFA
  - NFA
description: "Linz Chapter 2 in plain terms: deterministic and nondeterministic finite accepters, the extended transition on strings, and subset construction with small worked examples."
---

This continues from [Part I](/2026/11/06/automata-basics-for-coders-part-1/) (Linz and Rodger, [*An Introduction to Formal Languages and Automata*](https://www.betterreads.dev/discover/ia_an-introduction-to-formal-languages-and-automata-6th-edition?q=An%20Introduction%20to%20Formal%20Languages%20and%20Automata){:target="_blank" rel="noopener noreferrer"}, 7th ed., Chapter 1). Look up the book on BetterReads if you want the text beside these notes. Here the focus is **Chapter 2**: finite accepters.

Content:

1. Deterministic finite accepters (DFAs)
2. Transition graphs and tables
3. The extended transition on a string; acceptance
4. Nondeterministic finite accepters (NFAs)
5. Subset construction (NFA → DFA)
6. Cheat sheet and self-check

---

## Deterministic finite accepters

### Definition

A **deterministic finite accepter** (DFA) is a five-tuple

$$
M = (Q,\ \Sigma,\ \delta,\ q_0,\ F)
$$

where:

| Component | Meaning |
|-----------|---------|
| $Q$ | finite set of **states** |
| $\Sigma$ | finite **input alphabet** |
| $\delta: Q \times \Sigma \to Q$ | **transition function** (total) |
| $q_0 \in Q$ | **start** state |
| $F \subseteq Q$ | **final** (accepting) states |

$Q \times \Sigma$ is the [**Cartesian product**](/2026/11/06/automata-basics-for-coders-part-1/#cartesian-product-all-ordered-pairs): every pair $(q, a)$ with $q \in Q$ and $a \in \Sigma$. So $\delta: Q \times \Sigma \to Q$ means from each state, on each alphabet symbol, there is a next state.

**Deterministic** and **total** together mean: from every state, for every symbol in $\Sigma$, there is **exactly one** next state. There is exactly one start state. There are no $\lambda$-transitions (no move without reading a symbol).

### Transition graph and table

The same $\delta$ can be drawn as a directed graph or written as a table.

**Toy DFA $M_1$.** Alphabet $\Sigma = \{0,1\}$. Accept every string that **ends with** $1$.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> q0
    q0 --> q0: 0
    q0 --> q1: 1
    q1 --> q0: 0
    q1 --> q1: 1
    q1 --> [*]
</div>

| $\delta$ | 0 | 1 |
|----------|---|---|
| $q_0$ | $q_0$ | $q_1$ |
| $q_1$ | $q_0$ | $q_1$ |

Start: $q_0$. Final: $F = \{q_1\}$.

Reading: $q_0$ means “the string so far does not end in $1$” (including the empty prefix). $q_1$ means “the string so far ends in $1$”.

Check totality: each of $q_0,q_1$ has both a $0$-edge and a $1$-edge. One start. So $M_1$ is a DFA.

### Extended transition and acceptance

$\delta(q,a)$ is one step. For a whole string $w$, Linz uses the **extended** transition $\delta^*(q,w)$ (also written $\hat\delta$ or $\Delta$ in some notes): start at $q$, apply $\delta$ once per symbol of $w$ left to right.

$$
\begin{align*}
\delta^*(q,\ \lambda) &= q \\
\delta^*(q,\ wa) &= \delta\!\bigl(\delta^*(q,w),\ a\bigr)
\end{align*}
$$

$M$ **accepts** $w$ iff $\delta^*(q_0, w) \in F$.

The **language** of $M$ is

$$
L(M) = \{\, w \in \Sigma^*\ \mid\ \delta^*(q_0,w) \in F \,\}.
$$

**Worked runs on $M_1$:**

| $w$ | run | end | accept? |
|-----|-----|-----|---------|
| $\lambda$ | stay $q_0$ | $q_0$ | no |
| $0$ | $q_0 \xrightarrow{0} q_0$ | $q_0$ | no |
| $1$ | $q_0 \xrightarrow{1} q_1$ | $q_1$ | yes |
| $01$ | $q_0 \xrightarrow{0} q_0 \xrightarrow{1} q_1$ | $q_1$ | yes |
| $10$ | $q_0 \xrightarrow{1} q_1 \xrightarrow{0} q_0$ | $q_0$ | no |
| $101$ | $q_0 \to q_1 \to q_0 \to q_1$ | $q_1$ | yes |

So $L(M_1) = \{ w \in \{0,1\}^* \mid w \text{ ends with } 1 \}$.

---

## Nondeterministic finite accepters

### Definition

An **NFA** (Linz: nondeterministic finite accepter) allows:

1. several next states for the same (state, symbol) pair, and/or  
2. missing transitions (no next state), and/or  
3. more than one start state  

(depending on the exact formalisation; Linz also introduces **$\lambda$-transitions** in the same chapter—moves that consume no input. This post stays with ordinary labelled transitions first.)

Operationally: the machine may be in a **set** of states. On input $a$, take the union of all $a$-successors of every current state. If that union is empty, the run dies.

In the transition table, each cell is a **set**: $\{q_i\}$, $\{q_i,q_j\}$, or $\emptyset$.

### Toy NFA with a missing transition

**Toy $N_1$.** $\Sigma = \{a,b\}$.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> q0
    q0 --> q1: a
    q0 --> q0: b
    q1 --> q1: a
    q1 --> [*]
</div>

|  | $a$ | $b$ |
|--|-----|-----|
| $q_0$ | $\{q_1\}$ | $\{q_0\}$ |
| $q_1$ | $\{q_1\}$ | $\emptyset$ |

One start $q_0$. Final $F = \{q_1\}$. From $q_1$ there is no $b$-transition, so this is **not** a DFA. It is an NFA.

### String transition on a set of states

Write $\hat\Delta(S,w)$ for “start from every state in $S$; after reading $w$, which states are possible?”

```text
current ← S
for each symbol σ in w (left to right):
    next ← ∅
    for each q in current:
        add all states in δ(q, σ) to next
    current ← next
return current
```

If `current` becomes $\emptyset$, it stays $\emptyset$.

**On $N_1$:**

$\hat\Delta(\{q_0\},\, aba)$:

1. start $\{q_0\}$  
2. read $a$: $\{q_1\}$  
3. read $b$: from $q_1$ → $\emptyset$  
4. read $a$: from $\emptyset$ → $\emptyset$  

Result: $\emptyset$. The string is rejected (no path survives).

$\hat\Delta(\{q_0\},\, aa) = \{q_1\}$ → accepted, since $q_1 \in F$.

Acceptance for an NFA: $w$ is accepted iff $\hat\Delta(Q_0, w)$ intersects $F$ (here $Q_0 = \{q_0\}$).

---

## Why introduce NFAs

Some languages are awkward to design as DFAs by hand but easy as NFAs. A standard pattern: **strings that contain a fixed block as a substring**.

### Toy: contains `01`

Alphabet $\Sigma = \{0,1\}$. Language:

$$
L = \{\, w \in \{0,1\}^* \mid \text{`01' occurs as a contiguous substring of } w \,\}.
$$

Examples: $01 \in L$, $0011 \in L$, $10 \notin L$, $0 \notin L$, $0110 \in L$.

**NFA $N_2$** (contains-pattern shape):

- $q_0$: matched nothing of the block yet  
- $q_1$: matched the leading $0$  
- $q_2$: matched $01$ (accept)  

Prefix: from $q_0$, loop on every symbol (any junk before the block).  
Suffix: from $q_2$, loop on every symbol (any junk after).  
From $q_0$ on $0$ there are **two** moves: stay via the loop, and enter $q_1$. That branching is the nondeterminism.

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> q0
    q0 --> q0: 0
    q0 --> q0: 1
    q0 --> q1: 0
    q1 --> q2: 1
    q2 --> q2: 0
    q2 --> q2: 1
    q2 --> [*]
</div>

Partial table (only the edges drawn above):

|  | $0$ | $1$ |
|--|-----|-----|
| $q_0$ | $\{q_0,q_1\}$ | $\{q_0\}$ |
| $q_1$ | $\emptyset$ | $\{q_2\}$ |
| $q_2$ | $\{q_2\}$ | $\{q_2\}$ |

Same language as some DFA; the NFA is smaller to draw by hand.

**Related languages (different diagrams):**

| Language | Loop on start over $\Sigma$? | Loop on accept over $\Sigma$? |
|----------|------------------------------|-------------------------------|
| equals `01` only | no | no |
| ends with `01` | yes | no |
| contains `01` | yes | yes |

---

## Subset construction (NFA → equivalent DFA)

Linz §2.3: every NFA has an equivalent DFA (same language). The construction builds DFA states that are **sets of NFA states**.

### Algorithm

```text
DFA_start ← { NFA start }          # here {q0}
Worklist ← { DFA_start }

while Worklist is not empty:
    take a set S from Worklist
    for each symbol σ in Σ:
        T ← ⋃_{q ∈ S} δ(q, σ)
        record DFA edge S --σ--> T
        if T is new: add T to Worklist

A DFA state S is accepting iff S ∩ F_NFA ≠ ∅
```

Only sets that appear are kept (lazy / reachable subset construction).

### Full construction for $N_2$ (contains `01`)

Start: $A_0 = \{q_0\}$.

| From | on $0$ | on $1$ |
|------|--------|--------|
| $\{q_0\}$ | $\{q_0,q_1\}$ | $\{q_0\}$ |
| $\{q_0,q_1\}$ | $\{q_0,q_1\}$ | $\{q_0,q_2\}$ |
| $\{q_0,q_2\}$ | $\{q_0,q_1,q_2\}$ | $\{q_0,q_2\}$ |
| $\{q_0,q_1,q_2\}$ | $\{q_0,q_1,q_2\}$ | $\{q_0,q_2\}$ |

Details for the first new set:

- From $\{q_0\}$ on $0$: $\delta(q_0,0) = \{q_0,q_1\}$.  
- From $\{q_0,q_1\}$ on $1$: $\delta(q_0,1) \cup \delta(q_1,1) = \{q_0\} \cup \{q_2\} = \{q_0,q_2\}$.  
- From $\{q_0,q_2\}$ on $0$: $\{q_0,q_1\} \cup \{q_2\} = \{q_0,q_1,q_2\}$.

Accepting DFA states: those containing $q_2$, i.e. $\{q_0,q_2\}$ and $\{q_0,q_1,q_2\}$.

Rename for a drawn DFA (optional):

| name | set | accept? |
|------|-----|---------|
| $A$ | $\{q_0\}$ | no |
| $B$ | $\{q_0,q_1\}$ | no |
| $C$ | $\{q_0,q_2\}$ | yes |
| $D$ | $\{q_0,q_1,q_2\}$ | yes |

<div class="mermaid">
stateDiagram-v2
    direction LR
    [*] --> A
    A --> B: 0
    A --> A: 1
    B --> B: 0
    B --> C: 1
    C --> D: 0
    C --> C: 1
    D --> D: 0
    D --> C: 1
    C --> [*]
    D --> [*]
</div>

$L(\text{this DFA}) = L(N_2)$. The DFA has no nondeterminism: each cell is a single named state.

---

## Remarks

1. **Incomplete DFA tables.** If a drawn “DFA” omits an edge, under Linz’s total-$\delta$ definition it is not a DFA. As an NFA, that missing edge is $\emptyset$ and kills that branch.  
2. **$\lambda$-NFAs.** Linz allows transitions on $\lambda$. Removing them uses $\lambda$-closure before and after ordinary moves. That is the next layer after ordinary subset construction; not required for the toys above.  
3. **Equivalence.** DFAs and NFAs define the same class of languages (the regular languages). NFAs change design effort, not expressive power relative to DFAs.

---

## Cheat sheet

```text
DFA     one start; δ(q,a) is exactly one state for every q,a
NFA     δ(q,a) is a set (possibly empty or size ≥ 2); optional λ-moves in Linz
δ*      run a DFA from one state on a whole string
Δ̂(S,w)  run from a set of states on a whole string (NFA view)
L(M)    { w | the run from the start(s) on w ends in an accept state }

subset construction
  DFA state  = set of NFA states
  on symbol a: union of all a-successors
  accept     = set meets an NFA accept state
```

---

## Self-check

1. Is $M_1$ still a DFA if the edge $q_1 \xrightarrow{0} q_0$ is deleted? Why?  
2. Compute $\hat\Delta(\{q_0\}, bab)$ on $N_1$.  
3. For $N_2$, compute the DFA successor of $\{q_0,q_1\}$ on $0$ without looking at the table.  
4. Give one string in $L(N_2)$ and one string in $\{0,1\}^* \setminus L(N_2)$.  
5. In the subset DFA for $N_2$, why is $\{q_0,q_1\}$ not accepting?

### Answers

**1.** No. $\delta(q_1,0)$ would be undefined, so $\delta$ is not total. As an NFA the cell is $\emptyset$.

**2.** $\{q_0\} \xrightarrow{b} \{q_0\} \xrightarrow{a} \{q_1\} \xrightarrow{b} \emptyset$. Result $\emptyset$.

**3.** $\delta(q_0,0) \cup \delta(q_1,0) = \{q_0,q_1\} \cup \emptyset = \{q_0,q_1\}$.

**4.** In: e.g. $01$, $101$, $0010$. Out: e.g. $\lambda$, $0$, $1$, $10$, $110$.

**5.** $\{q_0,q_1\} \cap \{q_2\} = \emptyset$. The block `01` has not been completed on that set.

---

## References

- Linz, P. and Rodger, S. H. [*An Introduction to Formal Languages and Automata*](https://www.betterreads.dev/discover/ia_an-introduction-to-formal-languages-and-automata-6th-edition?q=An%20Introduction%20to%20Formal%20Languages%20and%20Automata){:target="_blank" rel="noopener noreferrer"}, 7th ed., Chapter 2. (BetterReads has a 6th-ed. copy to look up.)
- Previous post: [Automata Basics for Coders, Part I](/2026/11/06/automata-basics-for-coders-part-1/).
