---
title: Alignment
related:
  - RLHF
  - DPO
  - Instruction Tuning
  - Guardrails
---
**Alignment** is the effort to make model behavior match human intentions and product values—typically helpful, honest, and harmless in the intended context. Pretraining alone optimizes next-token likelihood; alignment layers reshape that into usable assistant behavior.

### Techniques
- **[Instruction tuning](/ai-glossary/instruction-tuning/) / SFT** — teach format and task following from demos
- **Preference optimization** — [RLHF](/ai-glossary/rlhf/), [DPO](/ai-glossary/dpo/), and variants from ranked feedback
- **Runtime [guardrails](/ai-glossary/guardrails/)** — filters, policies, and refusal logic around the model

### Why it matters
Alignment is continuous evaluation work, not a one-time checkbox. Distribution shift, jailbreaks, and new tools reopen gaps. Measure with red-team suites and product-specific rubrics; public chat vibes alone understate risk and overstate readiness.
