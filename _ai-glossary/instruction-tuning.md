---
title: Instruction Tuning
related:
  - Supervised Fine-Tuning
  - Alignment
  - System Prompt
---
**Instruction tuning** fine-tunes a base LM on diverse (instruction, response) pairs so it follows natural-language commands instead of only continuing raw text.

Instruct/chat checkpoints (e.g., “Qwen-Instruct”) are typically base models after instruction tuning, sometimes plus preference optimization. Domain CPT often starts from an instruct checkpoint, then adds light format SFT again after CPT.
