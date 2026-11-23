---
marp: true
theme: zlu
title: "{{CODE}} · ML Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Supervised learning, loss, splits, logistic regression.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Supervised learning → loss / gradients → honest splits → linear classifiers.

<div class="cover-parts">SUPERVISED · LOSS · SPLITS · LOGISTIC</div>

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
      <strong>Supervised vs unsupervised</strong>
      <span class="desc">Labels, objectives, what the code optimises</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Loss and gradients</strong>
      <span class="desc">What “training” actually updates</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Train / val / test and leakage</strong>
      <span class="desc">Honest evaluation under a brief</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Logistic regression and boundaries</strong>
      <span class="desc">Sigmoid, linear score, decision rule</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Supervised learning

## Labels, tasks, and what the model returns · ~15 min

---

# Supervised vs unsupervised

<p class="eyebrow">A1 · Task type</p>

| Kind | You are given | Goal |
|---|---|---|
| **Supervised** | inputs $x$ **and** labels $y$ | learn $f$ so $f(x)\approx y$ |
| **Unsupervised** | inputs $x$ only | find structure (clusters, factors, density) |
| **Reinforcement** | states, actions, rewards | learn a policy that maximises return |

Most coursework “ML” labs are supervised: classification ($y$ discrete) or regression ($y$ continuous).

<div class="tip">
<strong>Habit.</strong> Before coding, write one line: “predict ___ from ___.” If there is no target column, it is not a supervised fit.
</div>

---

# Loss: what training minimises

<p class="eyebrow">B1 · Objective</p>

Training picks parameters $\theta$ to make predictions close to labels on the **training** set.

$$
\theta^\star \approx \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n} \ell\!\bigl(y_i,\ f_\theta(x_i)\bigr)
$$

| Task | Common $\ell$ |
|---|---|
| Regression | squared error $(y-\hat y)^2$ |
| Binary class | logistic / binary cross-entropy |
| Multi-class | categorical cross-entropy |

A **gradient** $\nabla_\theta \ell$ points uphill. Gradient descent steps **opposite** that direction so $\theta$ lowers loss.

<div class="pitfall">
<strong>Trap.</strong> “Accuracy went up” is not the same as “loss went down” if the metric and the loss disagree (e.g. imbalanced classes).
</div>

---

# Train / validation / test

<p class="eyebrow">C1 · Honest splits</p>

| Split | Role |
|---|---|
| **Train** | fit $\theta$ |
| **Validation** | choose model / hyperparameters |
| **Test** | report once; do not retune |

**Leakage:** any information from val/test that reaches training (scaling fitted on all rows, target-encoded with future labels, random split on time series).

```text
fit scaler on train only → transform val/test
fit model on train → score val → lock choices → one test score
```

<div class="tip">
<strong>Brief check.</strong> If shuffling rows mixes “future” into train, the number is optimistic.
</div>

---

<!-- _class: section -->

# Part D · Logistic regression

## Linear score → probability → class · ~20 min

---

# Sigmoid and the decision rule

<p class="eyebrow">D1 · From score to class</p>

Linear score $z = w^\top x + b$. Probability via sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\hat p = \sigma(z),\qquad
\hat y = \mathbf{1}[\hat p \ge 0.5]
$$

The decision boundary is where $z=0$ (a hyperplane in $x$). Logistic regression is a **linear** classifier in feature space; nonlinearity comes only from features you build.

| Piece | Role |
|---|---|
| $w,b$ | parameters updated by training |
| $\sigma$ | squash score into $(0,1)$ |
| threshold | map probability to a label |

<div class="pitfall">
<strong>Exam trap.</strong> Changing the threshold changes precision/recall. It does not retrain $w$.
</div>
