---
title: DPO
synonyms:
  - Direct Preference Optimization
related:
  - RLHF
  - Alignment
  - Supervised Fine-Tuning
---
**Direct Preference Optimization (DPO)** aligns a model to preference pairs (chosen vs rejected responses) without training a separate reward model or running PPO. It reparameterizes the [RLHF](/ai-glossary/rlhf/) objective into a classification-style loss on the policy relative to a frozen reference model (usually the [SFT](/ai-glossary/supervised-fine-tuning/) checkpoint).

### How it works
Given a prompt and two completions, DPO raises the log-prob of the preferred answer and lowers that of the rejected one, with a KL-style term that keeps the policy near the reference. No online sampling loop is required during training—just offline preference data.

### When to use it
- After SFT, when you have ranked or pairwise human/AI preferences
- As a lighter alternative or complement to classic RLHF
- For style, safety, and helpfulness polish on domain assistants

Quality hinges on preference data noise and how far the reference already matches the target behavior. Over-strong DPO can overfit to judge quirks or collapse diversity.
