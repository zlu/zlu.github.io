---
marp: true
theme: zlu
title: "{{CODE}} · Systems Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Processes, sync, memory, sockets.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Processes and threads → sync → memory hierarchy → sockets.

<div class="cover-parts">PROCESS · SYNC · MEMORY · SOCKET</div>

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
      <strong>Process vs thread</strong>
      <span class="desc">Address spaces, shared memory, context switch</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Synchronisation basics</strong>
      <span class="desc">Critical sections, locks, races</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Virtual memory intuition</strong>
      <span class="desc">Pages, TLB, cache vs DRAM</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Sockets and HTTP labs</strong>
      <span class="desc">Byte streams, request / response</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Processes and threads

## Who shares what · ~15 min

---

# Process vs thread

<p class="eyebrow">A1 · Vocabulary</p>

| | Process | Thread (same process) |
|---|---|---|
| Address space | private | **shared** |
| Stack | own | own per thread |
| File descriptors / heap | process-wide | shared |
| Crash isolation | stronger | one bad thread can trash peers |

A **context switch** saves/restores CPU registers so another runnable entity can run. Threads switch cheaper than processes, but races are easier.

<div class="tip">
<strong>Habit.</strong> Before coding concurrency: list every shared variable. If two threads write it, you need a protocol.
</div>

---

# Critical section and race

<p class="eyebrow">B1 · Sync</p>

A **critical section** is code that touches shared state and must appear atomic to other threads.

**Race:** outcome depends on interleaving. Classic lost update:

```text
// both threads: balance = balance + 1
read balance → add 1 → write balance
```

Two increments can yield +1 if both read the same old value.

**Lock / mutex:** only one holder enters the section. Always unlock on every path (including errors).

<div class="pitfall">
<strong>Trap.</strong> “It passed once” only saw one interleaving. Stress tests and markers pick others.
</div>

---

# Virtual memory in one picture

<p class="eyebrow">C1 · Address translation</p>

CPU issues **virtual** addresses. MMU + page tables map them to **physical** frames.

| Idea | Plain meaning |
|---|---|
| Page | fixed-size chunk of virtual memory |
| Page fault | needed page not in RAM; OS loads / allocates |
| TLB | cache of recent translations |
| Locality | nearby / recent accesses hit cache and TLB |

Caches sit between CPU and DRAM. A lab that “randomly” walks a huge array often measures memory, not ALU speed.

---

<!-- _class: section -->

# Part D · Networking labs

## Bytes on a connection · ~20 min

---

# Socket + HTTP sketch

<p class="eyebrow">D1 · Layers you actually touch</p>

| Layer (lab view) | Job |
|---|---|
| Socket API | `connect` / `listen` / `read` / `write` byte streams |
| TCP | reliable ordered bytes (usually) |
| HTTP | text request line + headers + body |

Minimal HTTP request:

```text
GET /path HTTP/1.1
Host: example.com
<blank line>
```

Read until you have headers; then read `Content-Length` body bytes. Do not assume one `recv` returns the whole message.

<div class="pitfall">
<strong>Lab trap.</strong> Treating one `read()` as one application message. TCP is a stream; you must reassemble.
</div>
