---
title: Prompt Engineering
related:
  - Prompt Tuning
  - Chain-of-Thought
---
Prompt engineering is a manual, human-driven approach to designing effective prompts that elicit the desired output from a pre-trained language model. This method relies on understanding the behavior and limitations of the LLM and crafting input prompts accordingly. The process is akin to writing a clever query or instruction to get the best possible result without changing the underlying model parameters.

For example, consider a sentiment analysis task. A naive prompt might be:

"The movie was okay."

This may not give you a useful output unless you explicitly instruct the model. A prompt-engineered version would look like:

"Classify the sentiment of the following review as Positive, Negative, or Neutral: 'The movie was okay.'"

Prompt engineering involves iterations of trial-and-error, understanding model quirks, and using techniques like few-shot learning (giving examples in the prompt) or zero-shot learning (giving just the instruction) to guide the model
