---
marp: true
theme: zlu
title: zlu · Deck Template
description: Starter deck for the zlu Marp brand theme
paginate: true
size: 16:9
math: mathjax
html: true
footer: zlu
---

<!-- _class: cover -->

zlu · SESSION LABEL · PREP

# Your session title goes here

A short supporting line — what this session will do, and how you’ll walk it.

<div class="cover-parts">TOPIC A · TOPIC B · TOPIC C · TOPIC D</div>

---

<!-- _class: section -->

# Part A · Calculus

## Extrema & first-order ODEs · ~20 min

---

# Design tokens at a glance

| Token | Role |
|-------|------|
| Ink / Navy | Covers, titles, emphasis |
| Signal blue | Accent bar, badges, links |
| Mist / Paper | Content slide ground |
| Mist + left accent | Tip / pitfall / note callouts |

Fonts: **Sora** (display) + **Manrope** (body) + **IBM Plex Mono** (code). Brand mark: **zlu**.

---

# Roadmap pattern

Use HTML rows for timed agendas.

<div class="roadmap">
  <div class="roadmap-row">
    <div class="badge">A</div>
    <div>
      <strong>Calculus — extrema & ODE</strong>
      <span class="desc">Critical points, concavity, separation + IC</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Linear Algebra — eigenpairs</strong>
      <span class="desc">Characteristic polynomial, discriminant</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Probability — Bayes & stars-and-bars</strong>
      <span class="desc">Posterior vs sensitivity; counting configs</span>
    </div>
    <div class="meta">~25 min</div>
  </div>
</div>

---

# Two-column + shape cards

<p class="eyebrow">A1 · Core concepts</p>

<div class="columns">
<div>

## Critical point

A point $c$ is critical where $f'(c)=0$
(or $f'$ does not exist).

For a polynomial, just solve $f'(c)=0$.

</div>
<div>

## Concavity at a flat point

<div class="shape-cards">
  <div class="shape-card up">
    <div class="glyph">∪</div>
    <div class="eq">f″(c) &gt; 0</div>
    <p>cup · concave up · valley</p>
  </div>
  <div class="shape-card down">
    <div class="glyph">∩</div>
    <div class="eq">f″(c) &lt; 0</div>
    <p>cap · concave down · hill</p>
  </div>
</div>

</div>
</div>

---

<!-- _class: equation -->

# Separable ODE

If $y'=g(y)$ and $g(y)\neq 0$:

$$
\frac{dy}{g(y)}=dx
$$

Integrate both sides → apply the initial condition.

---

<!-- _class: tight -->

# Callouts for tutoring

<div class="note">
<strong>Write in English.</strong> Marks reward method steps, not only the final number.
</div>

<div class="tip">
<strong>Tip.</strong> After the general solution, plug in $y(x_0)=y_0$ before simplifying fancy forms.
</div>

<div class="pitfall">
<strong>Pitfall.</strong> Local max/min tests do not decide the global answer on $[a,b]$ — check endpoints.
</div>

---

# Table + badge

<p class="eyebrow">B2 · Discriminant</p>

<span class="badge-inline">D</span> Nature of eigenvalues for a real $2\times 2$:

| $D$ | Roots |
|-----|-------|
| $D>0$ | two distinct real eigenvalues |
| $D=0$ | one repeated real eigenvalue |
| $D<0$ | complex conjugate pair |

---

<!-- _class: end -->

# End of template

Copy a slide pattern above into your real deck.  
Keep `theme: zlu` and the math `$...$` convention.
