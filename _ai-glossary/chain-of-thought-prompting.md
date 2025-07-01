---
title: Chain of Thought Prompting
synonyms:
  - CoT
---
Definition: Chain-of-thought prompting is a method where the prompt includes intermediate reasoning steps, encouraging the model to “think out loud” and break down complex problems before answering.

LLMs like GPT can often solve simple problems directly. But for multi-step or reasoning-heavy tasks (e.g., math, logic puzzles, or common-sense reasoning), they perform significantly better when prompted to generate their reasoning first before concluding.

#### Example (with vs. without CoT)

**Question:**

If there are 3 cars and each car has 4 tires, how many tires are there in total?

**Without CoT:**

Prompt: “If there are 3 cars and each car has 4 tires, how many tires are there?”

Model Output: “4”

**With CoT:**

Prompt:

“If there are 3 cars and each car has 4 tires, how many tires are there?
Let’s think step by step.”

Model Output:

“There are 3 cars. Each car has 4 tires. So the total number of tires is 3 × 4 = 12.

Answer: 12”

####  Common CoT Prompts:
- “Let’s think step by step.”
- “First…, then…, so…”
- “Let me reason this out.”

#### Variants:
- Zero-shot CoT: Add only “Let’s think step by step.” to the prompt.
- Few-shot CoT: Include multiple worked examples with reasoning chains in the prompt.
- Automatic CoT: Generate reasoning steps automatically for many problems at scale.

Chain-of-thought prompting helps the model activate latent reasoning paths in its neural structure that are less likely to be triggered by short, direct prompts. It mimics how humans approach complex tasks: by breaking them down.
