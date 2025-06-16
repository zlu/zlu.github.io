---
layout: post
title: "Text Cluster Analysis: Grouping Documents by Similarity"
date: 2025-06-16
tags:
  - Text Cluster Analysis
  - R
  - KNN
description: "Text Cluster Analysis: Grouping Documents by Similarity"
comments: true
---

Happy Monday everyone!  The world is quite messy recently with the Middle East conflicts.  The chaotic nature may affect our perception of the world.  Like the French say "C'est la vie". But regardless of what the future may or may not bring to you, remember what the Swiss says, "La vie est belle".

Text cluster analysis groups similar documents based on their content, enabling automated categorization of large text collections in R.


## What is Text Cluster Analysis?

Cluster analysis groups documents so that those within the same cluster are more similar to each other than to those in other clusters. Similarity is often measured using distance metrics (e.g., Euclidean or cosine distance). It’s essential for organizing large document sets without manual intervention.

## Text Clustering Process

Text clustering involves five stages:
1. **Text Preprocessing**: Clean raw text by removing punctuation, numbers, stopwords, excess whitespace, and converting to lowercase. Apply stemming to reduce words to their root form.
2. **Term-Document Matrix (TDM)**: Represent documents as vectors of term frequencies.
3. **TF-IDF Normalization**: Weight terms by their importance (Term Frequency - Inverse Document Frequency) to emphasize unique terms and reduce the impact of common ones.
4. **Distance Computation and Clustering**: Compute distances between document vectors and apply a clustering algorithm (e.g., K-means, hierarchical, or HDBSCAN).
5. **Auto-Tagging**: Generate representative tags (cluster centers) for each cluster.

## Clustering Algorithms

### K-Means Clustering
K-means partitions documents into \( K \) clusters by:
- Selecting \( K \) seed documents.
- Assigning each document to the nearest seed based on Euclidean distance.
- Iteratively updating cluster centers.

**Example: K-Means Clustering**

```R
library(tm)
library(proxy)

# Preprocess text
# Suppose there are some text documents under "TextFile" directory
mytext <- DirSource("TextFile")
docs <- VCorpus(mytext)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stripWhitespace)

# Create TDM and apply TF-IDF
tdm <- DocumentTermMatrix(docs)
tdm.tfidf <- weightTfIdf(tdm)
tdm.tfidf <- removeSparseTerms(tdm.tfidf, 0.999)
tfidf.matrix <- as.matrix(tdm.tfidf)

# Compute cosine distance
dist.matrix <- dist(tfidf.matrix, method = "cosine")

# K-means clustering
kmeans <- kmeans(tfidf.matrix, centers = 3, iter.max = 100)
```

This code preprocesses text, creates a TF-IDF-weighted TDM, computes cosine distances, and applies K-means clustering with 3 clusters.

### Hierarchical Clustering
Hierarchical clustering builds a dendrogram by merging documents bottom-up based on similarity (e.g., complete, average, single, or Ward’s method).

### HDBSCAN (Hierarchical Density-Based Spatial Clustering)
HDBSCAN excels with:
- Arbitrarily shaped clusters.
- Varying cluster sizes and densities.
- Noise and outliers.
It identifies dense regions in data, separating clusters from sparse areas.

**Comparison with K-Means**:
- K-means assumes spherical clusters and struggles with noise.
- HDBSCAN adapts to complex cluster shapes and handles noise effectively.

## Text Clustering vs. Topic Modeling

| **Aspect**        | **Text Clustering**                          | **Topic Modeling**                          |
|-------------------|----------------------------------------------|---------------------------------------------|
| **Aim**           | Group documents into coherent categories     | Identify underlying themes in documents      |
| **Approach**      | Similarity measures (e.g., distance-based)   | Probabilistic distributions (e.g., LDA)      |
| **Output**        | Clusters with documents assigned to one      | Topics with associated word clusters        |

## Visualizing Clusters

Visualize clusters using principal components or an elbow plot to determine optimal \( K \):

```R
library(colorspace)
points <- cmdscale(dist.matrix, k = 2)  # Reduce to 2D
plot(points, main = "K-Means Clustering", col = as.factor(kmeans$cluster))

# Elbow plot for optimal K
cost_df <- data.frame()
for(i in 1:20) {
  kmeans <- kmeans(tfidf.matrix, centers = i, iter.max = 100)
  cost_df <- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}
names(cost_df) <- c("cluster", "cost")
plot(cost_df$cluster, cost_df$cost, type = "b")
```

The elbow plot helps identify the optimal number of clusters by showing where adding more clusters yields diminishing cost reductions.

Let's put everything together in a real-world scenario.

## Real-World Dataset Example: 20 Newsgroups

Before diving into the theory, let's explore a practical example using the 20 Newsgroups dataset, a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. This dataset is commonly used for text classification and clustering tasks.

```r
# Install required packages if not already installed
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(tm)) install.packages("tm")
if (!require(wordcloud)) install.packages("wordcloud")
if (!require(factoextra)) install.packages("factoextra")

# Load required libraries
library(tidyverse)
library(tm)
library(wordcloud)
library(factoextra)

# Create a sample Chinese news dataset in English translation
set.seed(123)
news_data <- data.frame(
  text = c(
    # Technology (科技)
    "Artificial intelligence technology is rapidly developing and changing our way of life.",
    "The popularization of 5G networks will promote widespread application of IoT technology.",
    "Quantum computing research has made major breakthroughs, greatly improving computing power.",
    "Blockchain technology has broad application prospects in the financial field.",
    
    # Sports (体育)
    "The Chinese women's volleyball team won gold again at the Olympics, bringing honor to the country.",
    "The NBA playoffs are in full swing with intense competition among teams.",
    "The World Cup qualifiers are about to begin, with national teams actively preparing.",
    "The new season of the Chinese Super League is about to kick off.",
    
    # Finance (财经)
    "The A-share market rose in volatile trading today, with the technology sector performing strongly.",
    "The central bank has introduced new policies to support the development of the real economy.",
    "The exchange rate of the Chinese yuan remains basically stable with sufficient foreign exchange reserves.",
    "China's economic growth is stable, with increasing vitality in the consumer market.",
    
    # Education (教育)
    "The Ministry of Education has introduced new policies to promote quality education reform.",
    "The university admissions process has begun, and candidates should be cautious when filling out applications.",
    "Online education platforms are developing rapidly, changing traditional learning methods.",
    "The revised Vocational Education Law has been passed, ushering in new development for vocational education."
  ),
  category = rep(c("Technology", "Sports", "Finance", "Education"), each = 4),
  stringsAsFactors = FALSE
)

# Custom stopwords
my_stopwords <- c("the", "and", "is", "in", "to", "of", "are", "with", "for", "as", "has", "have")

# Text preprocessing function
preprocess_text <- function(texts) {
  # Convert to lowercase
  texts <- tolower(texts)
  # Remove punctuation
  texts <- gsub("[[:punct:]]", "", texts)
  # Remove numbers
  texts <- gsub("[0-9]+", "", texts)
  # Remove extra whitespace
  texts <- gsub("\\s+", " ", trimws(texts))
  return(texts)
}

# Preprocess text
processed_texts <- preprocess_text(news_data$text)

# Create corpus
corpus <- VCorpus(VectorSource(processed_texts))

# Further text processing
processed_corpus <- corpus %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, my_stopwords) %>%
  tm_map(stripWhitespace)

# Create Document-Term Matrix
dtm <- DocumentTermMatrix(processed_corpus, control = list(
  wordLengths = c(3, Inf),  # Minimum word length of 3
  bounds = list(global = c(1, Inf))  # Terms must appear in at least 1 document
))

# Convert to matrix and apply TF-IDF
tfidf <- weightTfIdf(dtm)
tfidf_matrix <- as.matrix(tfidf)

# Perform K-means clustering (4 clusters for our 4 categories)
set.seed(123)
k <- 4
kmeans_result <- kmeans(tfidf_matrix, centers = k, nstart = 25)

# Add cluster assignments to original data
news_data$cluster <- kmeans_result$cluster

# Evaluate clustering against true categories
confusion_matrix <- table(Predicted = kmeans_result$cluster, 
                         Actual = as.factor(news_data$category))
print("Confusion Matrix:")
print(confusion_matrix)

# Visualize clusters using PCA
pca_result <- prcomp(tfidf_matrix, scale. = FALSE)
plot_data <- data.frame(
  PC1 = pca_result$x[,1], 
  PC2 = pca_result$x[,2],
  Cluster = as.factor(kmeans_result$cluster),
  Category = news_data$category
)

# Plot with improved visualization
ggplot(plot_data, aes(x = PC1, y = PC2, color = Cluster, shape = Category)) +
  geom_point(size = 4, alpha = 0.8) +
  theme_minimal() +
  labs(title = "K-means Clustering of News Articles",
       subtitle = "PCA Projection of TF-IDF Vectors",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme(legend.position = "bottom")

# Create word clouds for each cluster
par(mfrow = c(2, 2))
for (i in 1:k) {
  cluster_docs <- which(kmeans_result$cluster == i)
  cat("\nCluster", i, "contains", length(cluster_docs), "documents\n")
  
  # Get the documents in this cluster
  docs_in_cluster <- processed_corpus[cluster_docs]
  
  # Create a word cloud for the cluster
  wordcloud(docs_in_cluster, 
           max.words = 30, 
           random.order = FALSE,
           colors = brewer.pal(8, "Dark2"),
           scale = c(3, 0.5),
           main = paste("Cluster", i))
}
```
![text-clustering-word-cloud-zlu-me](/assets/images/uploads/text-clustering-word-cloud-zlu-me.png)

This example demonstrates how to:
1. Load and preprocess text data from the 20 Newsgroups dataset
2. Create a document-term matrix with TF-IDF weighting
3. Perform K-means clustering on the text data
4. Visualize the clusters using PCA
5. Create word clouds to explore the topics in each cluster

The confusion matrix shows how well the clustering matches the actual newsgroup categories, and the visualization will help you see how well-separated the clusters are in the reduced-dimensional space.

```R
[1] "Confusion Matrix:"
         Actual
Predicted Education Finance Sports Technology
        1         0       0      2          0
        2         0       0      1          0
        3         3       4      1          4
        4         1       0      0          0

Cluster 1 contains 2 documents
Cluster 2 contains 1 documents
Cluster 3 contains 12 documents
Cluster 4 contains 1 documents
```


## Conclusion

Text cluster analysis automates document categorization using preprocessing, TF-IDF, and algorithms like K-means or HDBSCAN. It’s ideal for organizing large text datasets, complementing topic modeling by focusing on document similarity rather than thematic content. Use R’s `tm` and `proxy` packages to implement and visualize clusters effectively.