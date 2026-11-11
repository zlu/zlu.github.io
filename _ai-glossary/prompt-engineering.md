---
title: Prompt Engineering
related:
  - Prompt Tuning
  - Chain-of-Thought
  - Prompt
  - System Prompt
  - Few-Shot
  - Zero-Shot
  - Soft Prompt
---
**Prompt engineering** is the craft of writing and structuring inputs so a frozen pretrained model does what you want—without updating weights. It is product logic expressed as text: roles, constraints, examples, and output formats.

Unlike [prompt tuning](/ai-glossary/prompt-tuning/) (learned [soft prompts](/ai-glossary/soft-prompt/)), prompt engineering edits readable [hard prompts](/ai-glossary/hard-prompt/) and [system prompts](/ai-glossary/system-prompt/). Iteration is empirical: change wording, measure task metrics, keep versioned templates.

### Common techniques
- Clear instructions and schemas (JSON keys, bullet rules)
- [Zero-shot](/ai-glossary/zero-shot/) vs [few-shot](/ai-glossary/few-shot/) exemplars
- [Chain-of-thought](/ai-glossary/chain-of-thought-prompting/) for multi-step reasoning
- Separating system policy from user content

### Example
Weak: `The movie was okay.`  
Stronger: `Classify the sentiment as Positive, Negative, or Neutral. Review: "The movie was okay."`

Prompt engineering does not replace retrieval, tools, or fine-tuning when the model lacks facts or format fidelity—but it is the cheapest first lever.
