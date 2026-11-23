---
marp: true
theme: zlu
title: "{{CODE}} · Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Spec reading, debug method, exam shapes.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Spec literacy → debug method → write-up claims → exam shapes.

<div class="cover-parts">SPEC · DEBUG · WRITE-UP · EXAM</div>

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
      <strong>Read the assessed brief</strong>
      <span class="desc">Inputs, outputs, rubric language</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Debug with a hypothesis</strong>
      <span class="desc">Reproduce → bisect → patch root cause</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Write-up that matches evidence</strong>
      <span class="desc">Claims tied to figures / tests</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Exam question shapes</strong>
      <span class="desc">Define, compute, compare, design</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Spec literacy

## What will be marked · ~15 min

---

# Brief → checklist

<p class="eyebrow">A1 · Method</p>

Extract before coding or memorising:

| Item | Write down |
|---|---|
| Deliverables | files, formats, due rule |
| Functional requirements | must / should |
| Non-goals | out of scope |
| Rubric verbs | explain, analyse, implement, evaluate |
| Hidden tests | edges implied by constraints |

<div class="tip">
<strong>Habit.</strong> Highlight every constraint number (sizes, time, precision). Those become tests.
</div>

---

# Debug loop

<p class="eyebrow">B1 · Discipline</p>

1. **Reproduce** with a minimal failing input
2. **Locate** — binary search prints / debugger / bisect commits
3. **Hypothesise** one cause; predict what you will see
4. **Patch** the cause; add a regression test
5. **Re-run** the original failing case and neighbours

| Anti-pattern | Replace with |
|---|---|
| Change three things at once | one change per experiment |
| “It works on my machine” | same command the marker runs |

<div class="pitfall">
<strong>Trap.</strong> Fixing a symptom (catch-all try/except) while the invariant stays broken.
</div>

---

# Claims need evidence

<p class="eyebrow">C1 · Write-ups</p>

Every claim in a report should point to a figure, table, test, or derivation.

```text
Claim → evidence → limitation
```

If the experiment did not measure it, do not assert it. Partial negative results are fine when stated clearly.

---

<!-- _class: section -->

# Part D · Exam shapes

## Pick a method per verb · ~15 min

---

# Question verb → first move

<p class="eyebrow">D1 · Map</p>

| Verb | First move on paper |
|---|---|
| Define | precise sentence + one tiny example |
| Compute / evaluate | write formula; substitute; box answer |
| Compare | table of criteria; then judgement |
| Design | constraints → candidates → choose → justify |
| Prove / show | goal line; pick proof pattern |

Timed drill: same shape twice, then wrong-answer review on the method, not only the number.

<div class="tip">
<strong>Keep.</strong> Spend 30 seconds classifying the verb before writing algebra or code.
</div>
