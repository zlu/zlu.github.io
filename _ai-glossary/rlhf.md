---
title: RLHF
synonyms:
  - Reinforcement Learning from Human Feedback
related:
  - DPO
  - Alignment
  - Supervised Fine-Tuning
  - Policy
  - Reinforcement Learning
---
**RLHF** trains a reward model from human preference comparisons, then optimizes the generative [policy](/ai-glossary/policy/) (the LLM) with [reinforcement learning](/ai-glossary/reinforcement-learning/) (often PPO) to maximize that reward while staying close to an [SFT](/ai-glossary/supervised-fine-tuning/) reference via a KL penalty.

### Pipeline sketch
1. SFT on demonstrations
2. Collect preferences → train reward model
3. RL update of the policy against the reward model

### Tradeoffs
Improves helpfulness and harmlessness when done well, but is complex, expensive, and sensitive to reward hacking. Many teams now use simpler preference methods like [DPO](/ai-glossary/dpo/) for a large share of [alignment](/ai-glossary/alignment/) gains, reserving full RLHF for cases that need online exploration.
