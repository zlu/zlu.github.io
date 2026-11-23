---
marp: true
theme: zlu
title: "{{CODE}} · Algorithms Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Complexity, ADTs, graphs, DP.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Complexity → core ADTs → graph search → DP patterns.

<div class="cover-parts">BIG-O · ADT · GRAPH · DP</div>

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
      <strong>Asymptotic complexity</strong>
      <span class="desc">Compare candidates on growth, not constants alone</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Arrays, lists, trees, heaps, hash</strong>
      <span class="desc">Cost of the operation the brief needs</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>BFS / DFS</strong>
      <span class="desc">Queue vs stack; what each discovers</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>DP pattern</strong>
      <span class="desc">State, transition, base, order</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Complexity

## Growth rates on paper · ~15 min

---

# Big-O in exam language

<p class="eyebrow">A1 · Vocabulary</p>

$f(n)=O(g(n))$ means: for large $n$, $f$ grows **no faster** than a constant times $g$.

| Class | Typical ops |
|---|---|
| $O(1)$ | array index, hash expected |
| $O(\log n)$ | balanced BST / binary search |
| $O(n)$ | single scan |
| $O(n\log n)$ | efficient comparison sort |
| $O(n^2)$ | nested scans of $n$ |

Drop lower terms and constant factors for asymptotics. Keep them when the brief asks for exact instruction counts.

<div class="tip">
<strong>Habit.</strong> Name $n$ first (items, edges, bits). Wrong $n$ → wrong answer.
</div>

---

# Pick the ADT by the operation

<p class="eyebrow">B1 · Structures</p>

| Need | Reach for |
|---|---|
| Index by position | array / vector |
| Insert/delete at ends | deque; middle → list / other |
| Min / max repeatedly | heap |
| Membership / key→value | hash table (avg), tree map (ordered) |
| Hierarchical / sorted order | BST / tree |

Always state average vs worst case when hashing appears.

<div class="pitfall">
<strong>Trap.</strong> Using a list for repeated “find min” is $O(n)$ per query; a heap is $O(\log n)$.
</div>

---

# BFS vs DFS

<p class="eyebrow">C1 · Graph search</p>

| | BFS | DFS |
|---|---|---|
| Frontier | **queue** | **stack** / recursion |
| Discovers | layer by distance | deep paths first |
| Unweighted shortest path | yes (edge count) | no (in general) |

Mark visited when you **enqueue** (BFS) or as the course specifies, so you do not explode the frontier.

```text
BFS: enqueue start; while queue: pop, expand unseen neighbours
```

---

<!-- _class: section -->

# Part D · Dynamic programming

## State machine on paper · ~20 min

---

# DP four-line template

<p class="eyebrow">D1 · Method</p>

1. **State** — what does `dp[...]` mean in one sentence?
2. **Transition** — how is a state built from smaller ones?
3. **Base** — smallest cases with known answers
4. **Order** — fill so dependencies already exist

Toy (1D): climbing stairs, $dp[i]=dp[i-1]+dp[i-2]$, $dp[0]=1, dp[1]=1$.

<div class="pitfall">
<strong>Exam trap.</strong> Writing code before naming the state. If you cannot say what `dp[i][j]` counts, the recurrence will be wrong.
</div>
