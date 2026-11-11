---
title: Reinforcement Learning
synonyms:
  - RL
related:
  - RLHF
  - Policy
  - DPO
  - Alignment
---
**Reinforcement learning (RL)** trains an agent to maximize expected reward through interaction with an environment. A [policy](/ai-glossary/policy/) maps states to actions; learning uses returns, advantages, and exploration–exploitation tradeoffs.

### LLM connection
In [RLHF](/ai-glossary/rlhf/) / RLAIF-style [alignment](/ai-glossary/alignment/), the environment is often a preference or reward model over generated text, and actions are tokens. Online RL (e.g. PPO) is powerful but operationally heavy; many teams get much of the gain from preference objectives like [DPO](/ai-glossary/dpo/) without a full RL loop.

### Contrast
Unlike supervised learning’s fixed labels, RL feedback can be sparse, delayed, and non-differentiable — which is why reward models and preference data matter so much for LLMs.
