---
marp: true
theme: zlu
title: "{{CODE}} · Automata Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). DFA, NFA, regex, Moore, Petri.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

DFA / NFA → regex → transducers → Petri nets.

<div class="cover-parts">DFA · NFA · REGEX · MOORE · PETRI</div>

---

# Roadmap · concept review

<style scoped>
.roadmap { gap: 0.28rem; margin-top: 0.35rem; }
.roadmap-row { padding: 0.38rem 0.7rem; gap: 0.7rem; }
.roadmap-row .badge { width: 1.9rem; height: 1.9rem; font-size: 0.85em; border-radius: 8px; }
.roadmap-row strong { font-size: 0.82em; margin-bottom: 0; }
.roadmap-row span.desc { font-size: 0.68em; }
.roadmap-row .meta { font-size: 0.72em; }
</style>

<div class="roadmap">
  <div class="roadmap-row">
    <div class="badge">A</div>
    <div>
      <strong>DFA five-tuple and $\hat\delta$</strong>
      <span class="desc">Read the figure; accept / reject</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>NFA and subset construction</strong>
      <span class="desc">Sets of states → equivalent DFA</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Regex membership</strong>
      <span class="desc">Parse operators; split into blocks</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Moore vs Petri fire</strong>
      <span class="desc">Output on states; marking enable/fire</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · DFA

## Read the figure · ~20 min

---

# DFA five-tuple

<p class="eyebrow">A1 · Vocabulary</p>

$$
M=(Q,\Sigma,\delta,q_0,F)
$$

| Piece | Meaning |
|---|---|
| $Q$ | finite states |
| $\Sigma$ | input alphabet |
| $\delta:Q\times\Sigma\to Q$ | total transition |
| $q_0$ | start |
| $F\subseteq Q$ | accepting states |

Extended transition $\hat\delta(q,\lambda)=q$, $\hat\delta(q,wa)=\delta(\hat\delta(q,w),a)$. Accept $w$ iff $\hat\delta(q_0,w)\in F$.

<div class="pitfall">
<strong>Trap.</strong> Empty string: stay at $q_0$; accept iff $q_0\in F$.
</div>

---

# NFA → DFA (subset)

<p class="eyebrow">B1 · Method</p>

DFA states = **reachable sets** of NFA states.

```text
start ← { nfa start }
for each reached set S and letter a:
  T ← ⋃_{q in S} δ(q,a)
  keep T if new
S accepting iff S hits an NFA accept state
```

NFA cells are sets (possibly empty). Same language power as DFA; design effort changes.

---

# Regex membership by splitting

<p class="eyebrow">C1 · Exam habit</p>

Parse precedence: `*` / `+` tightest, then concat, then `|`.

Cover the candidate string with blocks matching the top-level structure. Reject leftovers.

| Expression idea | Check |
|---|---|
| $r\|s$ | matches $r$ or $s$ |
| $rs$ | split into prefix / suffix |
| $r^*$ | zero or more repeats of $r$ |

---

<!-- _class: section -->

# Part D · Transducers and Petri

## Output modes and tokens · ~20 min

---

# Moore output vs Petri fire

<p class="eyebrow">D1 · Two models</p>

**Moore:** output $\lambda(q)$ on the **state**. On input length $n$, you emit $n+1$ outputs (includes start).

**Petri:** places hold tokens; a transition is **enabled** if every input place has enough tokens; **fire** consumes inputs and produces outputs atomically.

| Model | Exam first move |
|---|---|
| Moore from brief | list $\Sigma$, $\Gamma$, modes, then $\lambda$, then $\delta$ |
| Petri | write marking digits; list enabled; fire one |

<div class="tip">
<strong>Keep.</strong> Do not “see” mutual exclusion by intuition. Rewrite the marking after each fire.
</div>
