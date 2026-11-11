---
title: Policy
related:
  - Reinforcement Learning
  - RLHF
  - DPO
  - Sampling
  - Alignment
---
In [reinforcement learning](/ai-glossary/reinforcement-learning/), a **policy** $\pi(a \mid s)$ maps states to a distribution over actions. Learning improves expected return under that policy.

### In LLM alignment
The policy *is* the language model: the state is the prompt (+ partial completion), and actions are next-[token](/ai-glossary/token/) choices via [sampling](/ai-glossary/sampling/). [RLHF](/ai-glossary/rlhf/) optimizes this policy against a reward model, usually with a KL penalty toward a frozen reference (often the SFT model) so the policy does not drift into gibberish or reward hacking.

### Related
Preference methods like [DPO](/ai-glossary/dpo/) update the same policy directly from preference pairs without an explicit online RL loop, but the object being aligned is still the generative policy.
