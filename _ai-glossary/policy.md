---
title: Policy
related:
  - Reinforcement Learning
  - RLHF
---
In RL, a **policy** maps states to a distribution over actions. In LLM alignment, the policy is the language model itself — sampling tokens as actions.

RLHF optimizes this policy against a reward model with a KL penalty to a reference policy.
