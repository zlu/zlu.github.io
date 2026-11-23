---
marp: true
theme: zlu
title: "{{CODE}} · Programming Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Briefs, structure, recursion, language focus.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Read the brief → design → implement → test the edges.

<div class="cover-parts">{{COVER_PARTS}}</div>

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
      <strong>Brief → design before typing</strong>
      <span class="desc">Inputs, outputs, invariants, edge cases</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Functions and readable structure</strong>
      <span class="desc">One job per function; name the data</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Recursion checklist</strong>
      <span class="desc">Base case, recursive case, progress</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>{{ROADMAP_D_TITLE}}</strong>
      <span class="desc">{{ROADMAP_D_DESC}}</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · From brief to design

## What the marker will run · ~15 min

---

# Brief checklist (before code)

<p class="eyebrow">A1 · Spec literacy</p>

Write these four lines on paper first:

1. **Inputs** — types, ranges, null/empty rules
2. **Outputs** — exact format the autograder parses
3. **Invariants** — what must stay true after each step
4. **Edges** — empty, one element, max size, duplicates, negatives

| Symptom in lab | Usually missing |
|---|---|
| Works on sample, fails hidden | an edge from the brief |
| Wrong print format | output contract |
| Infinite loop / stack overflow | progress toward base case |

<div class="tip">
<strong>Habit.</strong> Restate the brief in one sentence you could explain in a viva. If you cannot, do not start coding.
</div>

---

# One job per function

<p class="eyebrow">B1 · Structure</p>

Split work so each function has a name that is a verb phrase and a single reason to change.

```text
parse_input → validate → core_algorithm → format_output
```

Keep I/O at the edges. Keep the algorithm pure when the language allows it. That makes unit tests cheap.

<div class="pitfall">
<strong>Trap.</strong> A 80-line `main` that mixes parsing, logic, and printing is hard to debug and hard to mark partial credit on.
</div>

---

# Recursion: three must-haves

<p class="eyebrow">C1 · Method</p>

Every correct recursive function needs:

| Piece | Question |
|---|---|
| **Base case** | When do I stop and return a concrete value? |
| **Recursive case** | How do I reduce to a smaller instance of the same problem? |
| **Progress** | Does each call move strictly closer to the base case? |

Toy: factorial $n! = n\cdot(n-1)!$ with $0!=1$.

Trace $n=3$ on paper: $3\to 2\to 1\to 0$ then multiply on the way back.

<div class="tip">
<strong>Debug.</strong> If it never ends, you are missing progress or the base case is unreachable.
</div>

---

<!-- _class: section -->

# {{PART_D_SECTION_TITLE}}

## {{PART_D_SECTION_SUB}} · ~20 min

---

{{PART_D_SLIDE}}
