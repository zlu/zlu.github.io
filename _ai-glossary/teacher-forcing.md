---
title: Teacher Forcing
synonyms:
  - Teacher-Forcing
related:
  - Decoder
  - Sequence-to-Sequence
  - Encoder–Decoder
  - Next-Token Prediction
  - Autoregressive
---
**Teacher forcing** is a training trick for models that generate one token at a time: at each step, feed the [decoder](/ai-glossary/decoder/) the *correct* previous token from the reference, not the token the model just predicted.

Without it, an early mistake becomes the next input and can cascade (exposure bias). With it, every step is a clean “given the right history so far, predict the next word” problem—exactly [next-token prediction](/ai-glossary/next-token-prediction/) under gold prefixes.

### Train vs inference
- **Train:** condition on gold tokens (teacher forcing)
- **Inference:** condition on the model’s own outputs ([autoregressive](/ai-glossary/autoregressive/) generation)

Scheduled sampling and similar methods partially close that gap by mixing gold and model tokens during training. Teacher forcing remains the default for seq2seq and causal LM training because it is stable and highly parallelizable on teacher prefixes.
