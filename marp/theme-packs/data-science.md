---
marp: true
theme: zlu
title: "{{CODE}} · Data Science Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Tables, EDA, features, visualisation claims.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Tidy tables → EDA → features → claims that match the plot.

<div class="cover-parts">TIDY · EDA · FEATURES · CLAIM</div>

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
      <strong>Tidy tables and joins</strong>
      <span class="desc">One observation per row; keys that match</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>EDA before modelling</strong>
      <span class="desc">Distributions, missingness, outliers</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Feature hygiene</strong>
      <span class="desc">Scale, encode, leakage checks</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Plots that support a claim</strong>
      <span class="desc">Match geometry to the question</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Tables

## Structure before clever code · ~15 min

---

# Tidy data rules

<p class="eyebrow">A1 · Layout</p>

| Rule | Meaning |
|---|---|
| Each variable | its own column |
| Each observation | its own row |
| Each value | its own cell |

Joins need a clear **key**. Inner join drops non-matches; left join keeps left rows and null-fills. Count rows before and after every join.

```text
n_before → join → n_after
assert n_after makes sense for the key cardinality
```

<div class="tip">
<strong>Habit.</strong> Print `shape`, dtypes, and a 3-row head after every major transform.
</div>

---

# EDA checklist

<p class="eyebrow">B1 · Look before you fit</p>

1. Missing rate per column
2. Univariate distribution (hist / value counts)
3. Impossible values (negative ages, future dates)
4. Duplicates on the entity key
5. Target balance (for classification)

| Finding | Typical next step |
|---|---|
| Many nulls | drop, impute, or model missingness |
| Heavy skew | transform or robust model |
| Class imbalance | metric choice, not only accuracy |

<div class="pitfall">
<strong>Trap.</strong> Dropping rows after the train/test split differently than on train leaks test information into cleaning policy.
</div>

---

# Feature hygiene

<p class="eyebrow">C1 · Before the model</p>

| Step | Fit on |
|---|---|
| Imputer / scaler | **train** only, then transform others |
| One-hot / target encode | train categories; unseen → explicit bucket |
| Train/val/test split | before any fit that sees $y$ or full $X$ |

Feature from “future” columns relative to prediction time = leakage.

---

<!-- _class: section -->

# Part D · Visual claims

## Geometry matches the question · ~15 min

---

# Choose the plot for the claim

<p class="eyebrow">D1 · Communication</p>

| Claim | Prefer |
|---|---|
| Distribution of one numeric | histogram / KDE / box |
| Count by category | bar |
| Relationship of two numerics | scatter |
| Change over time | line (ordered time axis) |

Title states the claim; axes named with units. Do not dual-axis two unrelated series to force a story.

<div class="tip">
<strong>Write-up.</strong> One sentence under the figure: what to see, and what you conclude.
</div>
