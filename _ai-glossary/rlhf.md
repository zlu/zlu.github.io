---
title: RLHF
synonyms:
  - Reinforcement Learning from Human Feedback
related:
  - DPO
  - Alignment
  - Supervised Fine-Tuning
---
**RLHF** trains a reward model from human preference comparisons, then optimizes the policy (the LLM) with reinforcement learning (often PPO) to maximize that reward while staying close to an SFT reference policy.

It improves helpfulness and harmlessness but is complex and expensive. Many teams now prefer simpler preference methods like DPO for a large fraction of alignment gains.
