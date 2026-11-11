---
title: LSTM
synonyms:
  - Long Short-Term Memory
related:
  - RNN
  - Transformer
  - Sequence Model
---
**LSTM** cells use gates (input, forget, output) to control information flow through a memory cell, mitigating vanishing gradients on long sequences versus vanilla [RNNs](/ai-glossary/rnn/). They were the workhorse [sequence model](/ai-glossary/sequence-model/) for pre-Transformer NLP (translation, speech, tagging).

### vs Transformers
[Transformers](/ai-glossary/transformer/) replace recurrence with [self-attention](/ai-glossary/self-attention/), enabling full parallel training over the sequence and better long-range mixing at scale. For most new LLM work, Transformers dominate; LSTMs remain in some speech, time-series, and edge settings where streaming recurrence or tiny footprints matter.

### Takeaway
Know LSTMs for lineage and interviews; default new language-model designs to Transformers unless latency, hardware, or streaming constraints argue otherwise. Bidirectional LSTMs (BiLSTM) were common encoders before BERT-style masked LMs took over.
