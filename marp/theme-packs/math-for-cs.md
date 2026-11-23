---
marp: true
theme: zlu
title: "{{CODE}} · Math for CS Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Linear algebra, probability, optimisation, discrete proofs.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Linear algebra objects → probability language → gradients → proof habits.

<div class="cover-parts">LINALG · PROB · OPT · PROOF</div>

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
      <strong>Vectors, matrices, eigenpairs</strong>
      <span class="desc">Objects used in ML and graphics</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Probability language</strong>
      <span class="desc">Events, conditionals, expectation</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Gradients and constraints</strong>
      <span class="desc">Optimisation intuition for coursework</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Discrete proof checklist</strong>
      <span class="desc">Direct, induction, contradiction</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Linear algebra

## Name the object · ~20 min

---

# Vectors and matrices in CS

<p class="eyebrow">A1 · Objects</p>

| Object | CS reading |
|---|---|
| Vector $x\in\mathbb{R}^n$ | feature list / coordinates |
| Matrix $A\in\mathbb{R}^{m\times n}$ | linear map $x\mapsto Ax$ |
| Dot product $x^\top y$ | similarity / projection piece |
| Eigenpair $Av=\lambda v$ | directions $A$ only scales |

Matrix–vector multiply: each output entry is a dot product of a row with $x$. Shape check first: $(m\times n)(n\times 1)=(m\times 1)$.

<div class="tip">
<strong>Habit.</strong> Write dimensions above every product before expanding algebra.
</div>

---

# Probability essentials

<p class="eyebrow">B1 · Language</p>

| Symbol | Meaning |
|---|---|
| $\Omega$ | sample space |
| $P(A)$ | probability of event $A$ |
| $P(A\mid B)$ | conditional: $P(A\cap B)/P(B)$ |
| $E[X]$ | expectation of random variable $X$ |

Bayes (finite form):

$$
P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}
$$

For discrete $X$ with pmf $p$, $E[X]=\sum x\,p(x)$. For continuous, integrate $x f(x)$.

<div class="pitfall">
<strong>Trap.</strong> Confusing $P(E\mid H)$ (likelihood) with $P(H\mid E)$ (posterior).
</div>

---

# Gradient descent sketch

<p class="eyebrow">C1 · Optimisation</p>

For smooth $f(\theta)$, gradient $\nabla f$ points to steepest ascent. Descent update:

$$
\theta \leftarrow \theta - \eta\,\nabla f(\theta)
$$

Step size $\eta$ too large → diverge; too small → crawl. Constrained problems add feasible-set projection or Lagrange structure when the course requires it.

---

<!-- _class: section -->

# Part D · Proofs

## Pick a method on purpose · ~15 min

---

# Proof checklist

<p class="eyebrow">D1 · Method map</p>

| Shape of claim | First move |
|---|---|
| “For all $n$, …” with recurrence | induction: base + inductive step |
| “If $P$ then $Q$” | assume $P$, derive $Q$ (direct) |
| “No …” / uniqueness | contradiction or contrapositive |
| Set equality | double inclusion |

Write the **goal** as the last line first, then fill the bridge. Name every quantified variable.

<div class="tip">
<strong>Induction.</strong> State $P(k)$ explicitly. The step must use $P(k)$, not a vague “similarly”.
</div>
