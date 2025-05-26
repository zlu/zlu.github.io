---
title: token
---
In Natural Language Processing (NLP), a token is a basic unit of text used for processing and analysis. It typically represents a word, subword, character, or symbol, depending on the tokenization strategy.

### Definition:

A token is a meaningful element extracted from raw text during tokenization, the process of breaking text into smaller pieces.

### Common Types of Tokens:

Token Type	Example for “I’m learning NLP!”
Word token	["I", "'m", "learning", "NLP", "!"]
Subword	["I", "'", "m", "learn", "##ing", "NLP", "!"] (e.g., BERT)
Character	["I", "'", "m", " ", "l", "e", "a", "r", "n", "i", "n", "g", " ", "N", "L", "P", "!"]


### Why Tokens Matter:
	•	Input to models: NLP models operate on sequences of tokens, not raw text.
	•	Efficiency: Tokenizing helps standardize and normalize text, aiding in tasks like classification, translation, and summarization.
	•	Vocabulary mapping: Tokens are converted to numerical IDs using a vocabulary (lookup table), enabling neural models to process them.

### Tokenization Example (Python + NLTK):

```python
from nltk.tokenize import word_tokenize

text = "I'm learning NLP!"
tokens = word_tokenize(text)
print(tokens)
# Output: ['I', "'m", 'learning', 'NLP', '!']
```


### Summary:

A token in NLP is a unit of text—often a word or subword—that forms the basis for downstream processing and modeling. Tokenization strategy varies depending on the language and model architecture.
