---
marp: true
theme: zlu
title: "{{CODE}} · Web / Mobile Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Client/server, state, lifecycle, HCI basics.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Client / server → state and routing → mobile lifecycle → HCI checks.

<div class="cover-parts">API · STATE · LIFECYCLE · HCI</div>

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
      <strong>Client vs server responsibilities</strong>
      <span class="desc">Who validates, who holds secrets</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>State, routing, APIs</strong>
      <span class="desc">URL, component state, REST-ish JSON</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Mobile lifecycle</strong>
      <span class="desc">Foreground, background, saved state</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Accessibility / HCI basics</strong>
      <span class="desc">Goals, tasks, evaluation sketches</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Client and server

## Trust and duties · ~15 min

---

# Who does what

<p class="eyebrow">A1 · Split</p>

| Concern | Client | Server |
|---|---|---|
| UX / rendering | yes | optional SSR |
| Input validation | nice UX | **authoritative** |
| Secrets / API keys | never in shipped app if avoidable | yes |
| Authorisation | UI hide only | enforce |

API sketch: `METHOD /resource`, JSON body, status codes (`200`, `201`, `400`, `401`, `404`, `500`). Idempotent methods matter for retries (`GET`, `PUT` vs blind `POST`).

<div class="tip">
<strong>Habit.</strong> Any check that protects data must run on the server.
</div>

---

# State vs URL

<p class="eyebrow">B1 · Navigation</p>

| Kind | Example | Survives refresh? |
|---|---|---|
| URL / route | `/users/42` | yes (if designed so) |
| Component state | open modal flag | usually no |
| Server session / DB | cart contents | yes |

Put shareable identity in the route. Keep ephemeral UI in local state. Sync remote state with clear loading / error / empty views.

<div class="pitfall">
<strong>Trap.</strong> Storing the only copy of critical data in client memory with no persistence plan.
</div>

---

# Mobile lifecycle (course level)

<p class="eyebrow">C1 · Platform constraints</p>

Apps move: created → foreground → background → killed.

| Event | Typical need |
|---|---|
| Pause / background | save drafts; stop sensors |
| Resume | refresh stale data |
| Process death | restore from saved instance / disk |

Battery, permissions, and offline mode are part of the brief, not polish.

---

<!-- _class: section -->

# Part D · HCI sketch

## Evaluate against a goal · ~15 min

---

# Task-centred evaluation

<p class="eyebrow">D1 · Method</p>

1. State the **user goal** in one sentence
2. List the **task steps** the UI requires
3. Note failures (error messages, dead ends)
4. Pick a method: heuristic walkthrough, usability test protocol, or A11y checklist (labels, contrast, focus)

<div class="tip">
<strong>Write-up.</strong> Report observed behaviour, not “users will love it”.
</div>
