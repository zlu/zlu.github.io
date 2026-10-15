---
layout: post
title: "Teaching a Small LLM Your Domain — Part 1: Overview"
description: "A practical series on specializing a 4B-class open model with continued pretraining (CPT), light format SFT, and a rented GPU — using public-domain text as the sandbox."
keywords: "domain LLM, continued pretraining, CPT, LoRA, supervised fine-tuning, SFT, open model specialization, Qwen, PEFT, fine-tuning"
date: 2026-10-12
comments: true
series: domain-llm
part: 1
sitemap: true
tags:
  - LLM
  - LoRA
  - Fine-tuning
  - CPT
  - series
glossary:
  - Continued Pretraining
  - LoRA
  - Supervised Fine-Tuning
  - PEFT
  - Causal Language Model
  - RAG
---

Small open models are good enough that “always call a frontier API” is no longer the only move. If you care about a **specific domain** — a library of books, a document archive, a product corpus you are allowed to train on — you can specialize a 4B-class [instruct](/ai-glossary/instruct-model/) checkpoint so it sounds like it has actually read *your* text.

This series walks through that recipe end to end. Public-domain literature is the running example. The same mechanics apply to any lawful text corpus. Terms like [CPT](/ai-glossary/continued-pretraining/), [LoRA](/ai-glossary/lora/), and [SFT](/ai-glossary/supervised-fine-tuning/) are defined in the [AI Glossary](/ai-glossary/).

## Series map

| Part | Topic | Status |
| --- | --- | --- |
| **1** | Overview: CPT vs SFT, failure modes, where cloud fits | **this post** |
| **2** | Continued pretraining with code (corpus → LoRA → Trainer) | [next](/2026/10/12/domain-llm-part2-continued-pretraining/) |
| **3** | Light format SFT (manners, not facts) | planned |
| **4** | Renting a GPU, durable checkpoints, cost | planned |
| **5** | Eval smoke tests vs base model | planned |

## The mental model

Three stages, in order:

1. **Base instruct model** — already speaks and follows chat format (e.g. Qwen3-4B Instruct).
2. **[CPT (continued pretraining)](/ai-glossary/continued-pretraining/)** — keep [next-token](/ai-glossary/next-token-prediction/) training on raw domain text so weights absorb style and facts.
3. **Light format [SFT](/ai-glossary/supervised-fine-tuning/)** — a small chat set so answers stay short and well-shaped. This is *not* where the library gets learned.

[Retrieval](/ai-glossary/rag/) at serve time still helps. CPT is not a substitute for looking up the right passage. It *is* what makes the model stop sounding like it only skimmed the open web.

```mermaid
flowchart LR
  A[Base instruct LLM] --> B[CPT on domain text]
  B --> C[Light format SFT]
  C --> D[Serve + optional retrieval]
```

## What not to do

**Do not expect SFT alone to inject a library.** Fine-tuning only on Q&A pairs mostly teaches answer *style*. The model will still invent plot points and blend editions.

**Do not treat a laptop smoke test as a domain train.** A few hundred documents, tiny LoRA, short context, and a few dozen steps prove that your *code* runs. They do not prove the corpus landed in the weights. I have seen the base model and the “trained” adapter give the same wrong literary fact.

**Do not confuse “I have a corpus file” with “I trained on it.”** Seventy thousand rows on disk mean nothing if the job only sees the first few hundred and stops early.

## Laptop vs cloud (one table)

| Knob | Laptop smoke | Real CPT baseline |
| --- | --- | --- |
| LoRA targets | `q_proj`, `v_proj` | attn + MLP |
| Rank | 8 | 16 |
| Sequence length | 512 | 1024–2048 |
| Steps | tens | ~1 epoch over the full corpus |
| Device | MPS / small GPU | rented CUDA (A40 / 4090 class) |

On Apple silicon, a “serious” LoRA config at long context can cost on the order of **minutes per optimizer step**. That is how a “quick 100-step job” turns into a multi-day fantasy. Shrink the job for local CI; rent a GPU for the epoch.

## Cost intuition

For tens of thousands of chapter-sized samples on a mid-tier cloud GPU, one CPT epoch is often **tens of dollars**, not hundreds — if you stop the pod when training finishes. Part 4 will cover volumes, SSH upload, and not leaving the meter running.

## How we will judge success

Fixed prompts, asked the same way to:

1. Base model (no adapter)
2. CPT (+ optional format SFT) adapter
3. Optional: adapter + retrieved passages

A classic literature trap: *Who did Heathcliff marry?* The answer is Isabella Linton. If your specialized model still denies the marriage after a full CPT pass, the train did not stick — or retrieval needs to carry the load.

Also keep an **out-of-domain** question. A good specialist should refuse or hedge when the text is not in the corpus.

## Next

**[Part 2 — Continued pretraining with code](/2026/10/12/domain-llm-part2-continued-pretraining/)** builds the JSONL corpus, attaches LoRA, and runs causal LM training with Hugging Face `Trainer`.
