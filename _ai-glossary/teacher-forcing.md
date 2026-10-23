---
title: Teacher Forcing
synonyms:
  - Teacher-Forcing
related:
  - Decoder
  - Sequence-to-Sequence
  - Encoder–Decoder
  - Next-Token Prediction
---
**Teacher forcing** is a training trick for models that generate one token at a time: at each step, feed the decoder the *correct* previous token from the reference, not the token the model just predicted.

Without it, an early mistake becomes the next input and can cascade. With it, every step is a clean “given the right history so far, predict the next word” problem. At inference there is no reference, so the model must consume its own outputs instead.
