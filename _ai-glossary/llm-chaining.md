---
title: LLM Chaining
---
LLM chaining is the process of connecting multiple calls to a language model — each with a specific purpose — so that the output of one step becomes the input to the next.

#### Common Use Case Example:

Task: Generate a well-researched blog post from a user-supplied topic.

Chain:
- Prompt 1: “Summarize the key points about ‘climate change and agriculture’.”
  - → Output: High-level bullet points.
- Prompt 2: “Expand each bullet point into a detailed paragraph.”
  - → Output: Full article body.
- Prompt 3: “Generate a title and meta description based on this article.”
  - → Output: SEO-friendly title + summary.

Each stage builds on the previous one.

#### Why Use LLM Chaining?
- Decomposes complex tasks into manageable steps.
- Improves accuracy by isolating reasoning from generation.
- Enables control over different stages (reasoning, formatting, summarizing, etc.).
- Supports modularity — you can reuse steps across tasks.

#### Variants:
- Sequential Chaining: Step-by-step flow, as described above.
- Conditional Chaining: Path depends on a decision made at runtime.
- Parallel Chaining: Multiple prompts run independently, then merged.
