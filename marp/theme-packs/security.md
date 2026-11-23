---
marp: true
theme: zlu
title: "{{CODE}} · Security Concept Review"
description: Tutoring concept deck for {{CODE}} ({{UNI}}). Threat models, crypto primitives, web patterns, secure coding.
paginate: true
size: 16:9
math: mathjax
html: true
footer: "{{FOOTER}}"
---

<!-- _class: cover -->

zlu · {{CODE}} · {{UNI}}

# {{TITLE}}

Threat model → crypto primitives → web patterns → secure coding habits.

<div class="cover-parts">THREAT · CRYPTO · WEB · CODE</div>

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
      <strong>Threat models and trust boundaries</strong>
      <span class="desc">Who can do what to which asset</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">B</div>
    <div>
      <strong>Crypto primitives at course depth</strong>
      <span class="desc">Hash, MAC, encrypt, sign — jobs differ</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">C</div>
    <div>
      <strong>Named web / network patterns</strong>
      <span class="desc">Injection, XSS, CSRF at syllabus level</span>
    </div>
    <div class="meta">~20 min</div>
  </div>
  <div class="roadmap-row">
    <div class="badge">D</div>
    <div>
      <strong>Secure coding habits</strong>
      <span class="desc">Validate, least privilege, fail closed</span>
    </div>
    <div class="meta">~15 min</div>
  </div>
</div>

---

<!-- _class: section -->

# Part A · Threat modelling

## Assets, adversaries, boundaries · ~15 min

---

# Threat model skeleton

<p class="eyebrow">A1 · Method</p>

Answer four questions before picking a control:

1. **Asset** — what must stay confidential / intact / available?
2. **Adversary** — remote user, malicious insider, malware on client?
3. **Trust boundary** — where does untrusted input enter?
4. **Failure** — what is the unacceptable outcome?

| Bad framing | Better |
|---|---|
| “Make it secure” | “Prevent network attacker from reading password hashes at rest” |

<div class="tip">
<strong>Habit.</strong> Draw the trust boundary on the diagram. Controls sit on the boundary.
</div>

---

# Primitive jobs (not interchangeable)

<p class="eyebrow">B1 · Crypto vocabulary</p>

| Primitive | Job |
|---|---|
| Hash | fixed digest; integrity fingerprint (alone ≠ auth) |
| MAC / HMAC | integrity **and** authenticity with a key |
| Symmetric encrypt | confidentiality with shared key |
| Public-key encrypt | confidentiality to a public key |
| Digital signature | authenticity / non-repudiation with private key |

Encryption without integrity can still be unsafe under active attackers. Coursework often asks which primitive matches which goal.

<div class="pitfall">
<strong>Trap.</strong> “Hash the password” without salt/KDF as if that were full credential storage design.
</div>

---

# Syllabus web patterns

<p class="eyebrow">C1 · Named failures</p>

| Pattern | Core idea |
|---|---|
| Injection (SQLi) | untrusted string becomes code/query structure |
| XSS | attacker script runs in victim’s browser origin |
| CSRF | browser sends victim cookies to a state-changing URL |
| Open redirect | untrusted URL parameter used as redirect target |

Defence sketch: parameterised queries, output encoding, SameSite/CSRF tokens, allowlists.

---

<!-- _class: section -->

# Part D · Coding habits

## Assignment-scale hygiene · ~15 min

---

# Secure coding defaults

<p class="eyebrow">D1 · Checklist</p>

| Habit | Why |
|---|---|
| Validate on the **server** | clients lie |
| Least privilege | breached component does less damage |
| Fail closed | deny when check errors |
| Secrets outside source | no keys in git |
| Log security events | without logging secrets |

For labs: fix the root trust-boundary bug; do not only filter one exploit string the sample used.

<div class="tip">
<strong>Viva.</strong> Explain what the attacker can still do after your patch.
</div>
