---
layout: post
title: "Topic Modeling with LDA: Uncovering Hidden Themes in Text - Father's Day Edition"
date: 2025-06-15
tags:
  - topic modeling
  - R
  - LDA
description: "Topic Modeling with LDA: Uncovering Hidden Themes in Text"
comments: true
---

Happy Sunday and Happy Father's Day everyone!  The rain has stopped and the sky is clear again.  Let's start the morning with some R!

First let's understand the concepts behind topic modeling.  Topic modeling is a powerful text mining technique to discover hidden themes in a collection of documents. In this blog, we will learn about the concepts behind **Latent Dirichlet Allocation (LDA)** and demonstrates its application using R.  "Dirichlet" refers to the Dirichlet distribution, named after the German mathmatician Johann Peter Gustav Lejeune Dirichlet (1805-1859).  he made significant contributions to number theory, analysis, and mathematical physics.  The word `latent` refers to hidden or unknown, as in latent variables.  In topic modeling, it means that the topics are not directly observed but are inferred from the patterns of word co-occurrences in the documents.

## What is Topic Modeling?

Topic modeling identifies groups of words (topics) that best represent the information in a document collection. It’s a statistical method used for data exploration, revealing patterns or categories in text data without human intervention. LDA, a popular topic modeling algorithm, assumes documents are mixtures of topics, and topics are mixtures of words.  By using topic modeling, we can quickly understand the textual data structure and identify categories embedded in the text.  For example, we can have the following topics and associated words:

- **Topic 1 (Runaways)**: Words like "servant," "reward," "run."
- **Topic 2 (Government)**: Words like "state," "law," "congress."

LDA is a specific method (algorithm) to model topics in text data.  LDA does so by assuming a specific probabilistic process for how documents are generated.  Such assumptions include mixture of topics and topic as word distributions.  The advantage of LDA is that it handles large text collections well and it can work without labeled data (unsupervised).  The disadvantage is that it is sensitive to the number of topics specified and it can be computationally expensive.

For example, we have as input a document:

```
Document: "cat play dog water food cat plant dog play cat"
```

LDA would perform the following steps:

1. Initialize: Randomly assign topics to each word
  - cat (P), play (P), dog (P), water (G), food (P),
  - cat (P), plant (G), dog (P), play (P), cat (P)
2. Iterative Process:
    - For each word, temporarily remove it and:
      - Look at the current topic assignments of other words
      - Calculate two probabilities:
        - How often words in this document are assigned to each topic
        - How often this word appears in each topic across all documents
      - Reassign the word to a topic based on these probabilities
3. After many iterations:
    - The algorithm converges to discover:
      - Some topics (distributions of words)
      - For each document, the mixture of topics

Example output:
```
Discovered Topics:
- Topic 1: cat (0.3), dog (0.25), play (0.2), food (0.15), fish (0.1)
- Topic 2: plant (0.3), water (0.25), soil (0.2), grow (0.15), sun (0.1)

Document Topic Mixture:
- Document 1: 70% Topic 1, 30% Topic 2
```

Here is a more realistic example using `topicmodels` and `tidytext`:

```R
library(topicmodels)
library(tm)
library(tidytext)
library(ggplot2)

# Create a corpus with clear topics for demo purpose :)
docs <- c("machine learning algorithms require training data to make predictions",
          "deep neural networks are powerful for image recognition tasks",
          "gardening requires good soil, water, and sunlight for plant growth",
          "roses and tulips are beautiful flowers that need regular watering",
          "machine learning models can be trained using gradient descent",
          "watering plants in the morning helps prevent fungal diseases")

# Create Document-Term Matrix (DTM) 
corpus <- VCorpus(VectorSource(docs))
dtm <- DocumentTermMatrix(corpus, 
                         control = list(tolower = TRUE,
                                      removePunctuation = TRUE,
                                      removeNumbers = TRUE,
                                      stopwords = TRUE))

# Remove sparse terms
dtm <- removeSparseTerms(dtm, 0.7)

# Run LDA with 2 topics
set.seed(123)
lda_model <- LDA(dtm, k = 2, control = list(seed = 123))

# Extract and display topics with probabilities
topics <- tidy(lda_model, matrix = "beta")

# Show top 5 terms per topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Visualize top terms
top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Top Terms in Each Topic",
       x = "Probability (Beta)",
       y = "Term") +
  theme_minimal()

# Document-topic mixtures
doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  group_by(document) %>%
  top_n(1, gamma) %>%
  ungroup()

# Print document-topic assignments
doc_topics
```

And the output is:
```R
# A tibble: 10 × 3
   topic term        beta
   <int> <chr>      <dbl>
 1     1 watering  0.0509
 2     1 learning  0.0495
 3     1 helps     0.0428
 4     1 make      0.0420
 5     1 morning   0.0402
 6     2 machine   0.0574
 7     2 can       0.0412
 8     2 beautiful 0.0401
 9     2 learning  0.0394
10     2 gardening 0.0381
# A tibble: 6 × 3
  document topic gamma
  <chr>    <int> <dbl>
1 1            1 0.514
2 3            1 0.505
3 6            1 0.510
4 2            2 0.509
5 4            2 0.507
6 5            2 0.514
```

![topic-modeling-lda](/assets/images/uploads/tm.png)

The first tibble is the top terms per topic:
- The model identified 2 topics (Topic 1 and Topic 2)
- Topic 1 seems to be about "gardening/plant care" (top terms: watering, helps, morning)
- Topic 2 seems to be about "machine learning" (top terms: machine, learning, gardening)
- `beta` values represent the probability of the word appearing in that topic (higher = more representative)
- Note: The topics aren't perfectly separated because:
  - The word "learning" appears in both topics
  - The sample size is very small (only 6 documents)

The second tibble is the document-topic mixtures:
- It shows which topic is most likely for each document
- `gamma` values are very close to 0.5 for all documents, meaning:
  - The model is uncertain about topic assignments
  - Documents aren't strongly associated with either topic
  - This often happens with small datasets or when topics aren't well-separated

We can improve the model by using more documents and longer documents, or we can use a larger and more diverse dataset.

This shows us the discovered topics and how each document is a mixture of these topics, effectively `reversing` the document generation process we started with.  The key insight is that while the forward process generates documents from topics, LDA works backwards from the documents to disover the underlying topics and their mixtures.


## Visualizing Topic Modeling Using Real World DataSet

In our final example, we will visualize the Associated Press topics using ggplot2.  AssociatedPress dataset is baked into R, so we don't need to download it.  It contains 2,246 news articles from AP.  The articles have be pre-processed into DTM which each row represents a document and each column represents a term.  The values are term frequencies:

```R
<<DocumentTermMatrix (documents: 2246, terms: 10473)>>
Non-/sparse entries: 302031/23220327
Sparsity           : 99%
Maximal term length: 18
Weighting          : term frequency (tf)
```

### Steps:
1. Preprocess data: Convert to lowercase, remove symbols, punctuation, and stopwords.
2. Create a Document-Term Matrix (DTM).
3. Run LDA to extract topics.
4. Visualize results.

### Example: LDA on Associated Press Data

```R
library(topicmodels)
library(tidytext)
library(tidyr)
library(ggplot2)
library(dplyr)

data("AssociatedPress")
set.seed(1234)
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))  # 2-topic LDA
ap_topics <- tidy(ap_lda, matrix = "beta")  # Per-topic word probabilities

# Top 10 terms per topic
ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Visualize top terms
ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~topic, scales = "free") +
  coord_flip()
```

![associated-press-topic-modeling](/assets/images/uploads/tm-associated-press.png)

This code creates a two-topic LDA model, extracts word probabilities (`beta`), and visualizes the top 10 terms per topic. For example, Topic 1 may include "percent," "million" (business news), while Topic 2 includes "president," "government" (political news).

### Comparing Topics

To highlight differences between topics, compute the log ratio of word probabilities:

```R
beta_spread <- ap_topics %>%
  mutate(topic = paste0("topic", topic)) %>%
  spread(topic, beta) %>%
  filter(topic1 > 0.001 | topic2 > 0.001) %>%
  mutate(log_ratio = log2(topic2 / topic1))

beta_spread %>%
  mutate(term = reorder(term, log_ratio)) %>%
  ggplot(aes(term, log_ratio)) +
  geom_col(show.legend = FALSE) +
  coord_flip()
```

This visualizes terms with the greatest difference in probability between topics.

### Document-Topic Probabilities

Extract the proportion of each topic in each document:

```R
ap_documents <- tidy(ap_lda, matrix = "gamma")  # Per-document topic probabilities
```

### Inspecting Document Words

Check frequent words in a specific document:

```R
tidy(AssociatedPress) %>%
  filter(document == 6) %>%
  arrange(desc(count))
```

This might show words like "noriega" and "panama," indicating the document’s content.

## Advantages of LDA

Unlike hard clustering, LDA allows topics to share words (e.g., "people" in both business and political topics), reflecting natural language overlap. It’s ideal for qualitative analysis, such as studying trends in literature or social media.

## Conclusion

LDA is a robust method for uncovering latent themes in text data. Using R’s `topicmodels` and `tidytext` packages, you can preprocess text, apply LDA, and visualize results to gain insights into document collections. Experiment with your own data to discover hidden patterns!