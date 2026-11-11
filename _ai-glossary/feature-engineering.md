---
title: Feature Engineering
related:
  - Machine Learning
  - Representation Learning
  - Independent Variable
  - Embedding
---
**Feature engineering** constructs input variables that help classical ML models learn—scaling, encodings, interactions, domain aggregates, and leak-free time windows. Historically this was most of applied ML craft.

### Deep learning shift
[Representation learning](/ai-glossary/representation-learning/) moves much of that work into learned [embeddings](/ai-glossary/embedding/) and end-to-end nets. For text and vision, pretrained models often beat hand-crafted features. Tabular problems and hybrid systems still gain from careful features, missingness flags, and calibrated categoricals.

### Still useful in LLM stacks
- Metadata filters and structured fields for [hybrid search](/ai-glossary/hybrid-search/)
- Chunk titles, section paths, and entity tags as retrieval signals
- Tool schemas and typed slots that constrain generation

Good features reduce sample complexity; bad ones leak labels or encode outdated business rules—version them like code.
