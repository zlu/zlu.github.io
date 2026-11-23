---
marp: true
theme: zlu
title: "{{CODE}} · Databases Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). ER, SQL, indexes, transactions.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

ER design → SQL queries → indexes / plans → transactions.

<div class="cover-parts">ER · SQL · INDEX · TXN</div>

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
      <strong>ER and normalisation</strong>
      <span class="desc">Entities, keys, redundancy trade-offs</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>SQL joins and aggregation</strong>
      <span class="desc">Filter → join → group → having</span>
    </div>
    <div class="meta">~25 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Indexes and slow queries</strong>
      <span class="desc">What an index can and cannot fix</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Transactions intro</strong>
      <span class="desc">ACID and isolation in coursework terms</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Modelling

## Keys and redundancy · ~15 min

---

# ER essentials

<p class="eyebrow">A1 · Design</p>

| Concept | Meaning |
|---|---|
| Entity | thing you store rows about |
| Attribute | column |
| Relationship | association; note cardinality 1:1, 1:N, M:N |
| Primary key | unique row id |
| Foreign key | reference to another table’s key |

**Normalisation** reduces redundancy (and update anomalies). Denormalise only with a measured read-path reason.

<div class="tip">
<strong>Habit.</strong> Write cardinality on every relationship before creating tables.
</div>

---

# SQL mental order

<p class="eyebrow">B1 · Query shape</p>

Logical processing (simplified):

```text
FROM / JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

| Need | Pattern |
|---|---|
| Combine tables | `JOIN ... ON` keys |
| Per-group stats | `GROUP BY` + `COUNT/SUM/...` |
| Filter groups | `HAVING` (after aggregate) |
| Filter rows | `WHERE` (before aggregate) |

Nested query: use when a value or set must be computed first; prefer joins when both sides are tables you want to combine.

<div class="pitfall">
<strong>Trap.</strong> Putting aggregate filters in `WHERE` instead of `HAVING`.
</div>

---

# Why is this slow?

<p class="eyebrow">C1 · Indexes</p>

An **index** is a side structure (often B-tree) that finds rows by key without scanning the whole table.

Helps: equality / range on leading index columns, joins on indexed keys.
Hurts / ignores: function on the column (`WHERE YEAR(ts)=2024`), leading-wildcard `LIKE '%x'`, tiny tables, write-heavy loads (indexes cost on insert).

Read `EXPLAIN` / query plan: look for sequential scans on large tables when you expected an index.

---

<!-- _class: section -->

# Part D · Transactions

## All-or-nothing under concurrency · ~15 min

---

# ACID in one slide

<p class="eyebrow">D1 · Intro</p>

| Letter | Plain meaning |
|---|---|
| **A**tomicity | all statements commit, or none |
| **C**onsistency | constraints hold after commit |
| **I**solation | concurrent txns do not see each other’s dirty state (per level) |
| **D**urability | committed data survives crash |

Isolation levels trade anomaly freedom vs throughput. Coursework often asks which anomaly (dirty read, non-repeatable, phantom) a level allows.

<div class="tip">
<strong>Lab.</strong> Keep transactions short; do not hold locks while waiting on the network.
</div>
