---
title: Hallucination
---
The generation of output by a model that is not grounded in the input data or real-world facts.

#### Types of Hallucinations:

1. Factual Hallucination
The model generates information that is factually incorrect, even though it may sound plausible.

  Example: Saying “The Eiffel Tower is in Berlin.”

2. Faithfulness Hallucination
The model’s output does not accurately reflect or contradicts the input, especially common in summarization tasks.

  Example: Summarizing a paragraph to include details not present in the original text.

3. Mode Collapse or Memorized Hallucination
The model repeats phrases or inserts memorized content that is irrelevant or unrelated.

#### Why It Happens
- Overgeneralization from training data.
- Poor alignment with source input.
- Incomplete training data or biases.
- Lack of mechanisms for fact-checking or external grounding.

#### Mitigation Techniques
- Retrieval-augmented generation (RAG).
- Fact-checking pipelines.
- Reinforcement learning from human feedback (RLHF).
- Prompt engineering and input constraints.
