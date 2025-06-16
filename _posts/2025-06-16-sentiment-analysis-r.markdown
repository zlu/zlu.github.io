---
layout: post
title: "Sentiment Analysis: Decoding Emotions in Text"
date: 2025-06-16
tags:
  - Sentiment Analysis
  - R
description: "Sentiment Analysis: Decoding Emotions in Text"
comments: true
---

Built upon the blogs about text clustering and text model, we are ready to dive into a classic topic - Sentiment Analysis. Sentiment analysis computationally identifies and categorizes emotions in text, helping understand public opinions or consumer feedback.

## What is Sentiment Analysis?

Sentiment analysis determines the emotional tone behind text, classifying it as positive, negative, or neutral. It’s widely used for social media monitoring and understanding consumer needs.

### Why Use Sentiment Analysis?
- **Public Opinion**: Gauge sentiment on topics or brands.
- **Consumer Insights**: Quickly identify customer reactions (e.g., Expedia Canada’s commercial case).

### Challenges
Human language is complex, and machines struggle with nuances like sarcasm (e.g., “Verrryyyyyy goooodddd!!” may be misread as positive). Algorithms are evolving to handle such cases but aren’t 100% accurate.

## Sentiment Analysis Process

1. **Text Preprocessing**:
   - **Tokenization**: Split text into words or phrases.
   - **Stop Word Filtering**: Remove common words (e.g., “and,” “the”).
   - **Negation Handling**: Address negations (e.g., “not good” vs. “not not good”).
   - **Stemming**: Reduce words to their root form (e.g., “running” to “run”).
2. **Sentiment Classification**: Assign polarity (positive/negative) using lexicons or algorithms.
3. **Sentiment Scoring**: Quantify sentiment strength, considering factors like capitalization (e.g., “GOOD” indicates stronger emotion).

### Example Data
| Text | Sentiment |
|------|----------|
| Loves the German bakeries in Sydney... | Positive |
| @VivaLaLauren Mine is broken too!... | Negative |
| @Mofette briliant! May the fourth be with you... | Positive |

## Sentiment Analysis in R

Using R’s `tm`, `syuzhet`, and other packages, we can preprocess text and analyze sentiment.

### Preprocessing and Word Cloud

```R
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)

# Sample text data for sentiment analysis
text <- c(
  "I absolutely love this product! It has exceeded all my expectations and works perfectly.",
  "The service was terrible. I've never been more disappointed with a purchase in my life.",
  "This is just okay. Not great, but not bad either. It gets the job done I suppose.",
  "The customer support team was incredibly helpful and resolved my issue within minutes. Amazing!",
  "The quality is quite poor for the price. I expected much better based on the reviews.",
  "This is hands down the best thing I've bought this year. Worth every penny!",
  "I'm really frustrated with the shipping delay. The product is good but the wait was unacceptable.",
  "The instructions were unclear, but once I figured it out, the product worked as described.",
  "I wouldn't recommend this to anyone. Complete waste of money and time.",
  "The design is beautiful and it's very easy to use. I'm extremely satisfied with my purchase!"
)

docs <- Corpus(VectorSource(text))
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)
docs <- tm_map(docs, stemDocument)

# Create Term-Document Matrix
dtm <- TermDocumentMatrix(docs)
dtm_m <- as.matrix(dtm)
dtm_v <- sort(rowSums(dtm_m), decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v), freq=dtm_v)

# Generate word cloud with adjusted parameters
# Set up plot margins (bottom, left, top, right)
par(mar = c(0, 0, 0, 0))  # Remove all margins

# Create a new plot with larger dimensions
png("wordcloud.png", width = 10, height = 8, units = "in", res = 300)  # High resolution

set.seed(1234)
wordcloud(
  words = dtm_d$word, 
  freq = dtm_d$freq, 
  min.freq = 1,
  max.words = 50,
  random.order = FALSE, 
  rot.per = 0,            # No rotation
  scale = c(4, 0.8),      # Scale between largest and smallest words
  colors = brewer.pal(8, "Dark2"),
  vfont = c("sans serif", "plain"),
  use.r.layout = TRUE     # Better layout algorithm
)

dev.off()  # Close the device

# Display the saved image
if (requireNamespace("png", quietly = TRUE) && requireNamespace("grid", quietly = TRUE)) {
  library(png)
  library(grid)
  if (file.exists("wordcloud.png")) {
    img <- png::readPNG("wordcloud.png")
    grid::grid.raster(img)
  } else {
    warning("Word cloud image not found. Please check the file path.")
  }
} else {
  warning("Please install 'png' and 'grid' packages to display the word cloud.")
}
```
![Word Cloud](/assets/images/uploads/sentiment-analysis-word-cloud-zlu-me.png)
This code preprocesses text, removes noise, and visualizes frequent words in a word cloud.

### Sentiment Scoring

Using `syuzhet` for sentiment analysis with different lexicons:

```R
library(syuzhet)
library(ggplot2)

# Sentiment scoring with multiple methods
syuzhet_vector <- get_sentiment(text, method="syuzhet")
bing_vector <- get_sentiment(text, method="bing")
afinn_vector <- get_sentiment(text, method="afinn")

# Compare first few scores
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)

# Emotion classification with NRC
d <- get_nrc_sentiment(text)
td <- data.frame(t(d))
td_new <- data.frame(rowSums(td))
names(td_new) <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)

# Create a more informative plot
ggplot(td_new, aes(x = reorder(sentiment, count), y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +  # Remove legend since it's redundant
  labs(title = "Sentiment Analysis Results",
       x = "Sentiment",
       y = "Count") +
  scale_fill_brewer(palette = "Set3") +
  coord_flip()  # Flip coordinates for better readability

# Create multiple visualizations
# 1. Basic sentiment scores comparison
sentiment_scores <- data.frame(
  Text = 1:length(text),
  Syuzhet = syuzhet_vector,
  Bing = bing_vector,
  Afinn = afinn_vector
)

# Reshape data for plotting
sentiment_long <- tidyr::pivot_longer(sentiment_scores, 
                                    cols = c(Syuzhet, Bing, Afinn),
                                    names_to = "Method",
                                    values_to = "Score")

# Plot 1: Compare different sentiment scoring methods
p1 <- ggplot(sentiment_long, aes(x = Text, y = Score, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Comparison of Sentiment Scoring Methods",
       x = "Text Sample",
       y = "Sentiment Score") +
  scale_fill_brewer(palette = "Set2")

# Plot 2: NRC Emotion Analysis (already created above)
p2 <- ggplot(td_new, aes(x = reorder(sentiment, count), y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  labs(title = "Emotion Analysis Results",
       x = "Emotion",
       y = "Count") +
  scale_fill_brewer(palette = "Set3") +
  coord_flip()

# Plot 3: Word Cloud (already created above)
# The word cloud is saved as "wordcloud.png"

# Display all plots
print(p1)
print(p2)

# Print summary statistics
cat("\nSummary of Sentiment Scores:\n")
print(summary(sentiment_scores[, -1]))

# Print the most positive and negative texts
cat("\nMost Positive Text:\n")
print(text[which.max(syuzhet_vector)])
cat("\nMost Negative Text:\n")
print(text[which.min(syuzhet_vector)])
```

![Sentiment Scoring](/assets/images/uploads/sentiment-score-zlu-me.png)


![Sentiment Scoring](/assets/images/uploads/emotion-analysis-zlu-me.png)



```R

Summary of Sentiment Scores:
    Syuzhet            Bing           Afinn      
 Min.   :-1.750   Min.   :-2.00   Min.   :-5.00  
 1st Qu.:-0.250   1st Qu.: 0.00   1st Qu.:-0.75  
 Median : 0.325   Median : 0.00   Median : 1.50  
 Mean   : 0.600   Mean   : 0.80   Mean   : 2.20  
 3rd Qu.: 1.738   3rd Qu.: 2.75   3rd Qu.: 5.75  
 Max.   : 3.150   Max.   : 4.00   Max.   :10.00  

Most Positive Text:
[1] "The customer support team was incredibly helpful and resolved my issue within minutes. Amazing!"

Most Negative Text:
[1] "I'm really frustrated with the shipping delay. The product is good but the wait was unacceptable."
```



This code applies sentiment scoring using `syuzhet`, `bing`, and `afinn` lexicons and visualizes emotions (e.g., joy, sadness) with the NRC lexicon.

## Lexicon-Based Analysis

Lexicons like `bing` and `afinn` assign sentiment scores to words:
- **Bing**: Binary (positive/negative, e.g., “abandon” = negative).
- **Afinn**: Numeric scores (e.g., “abandon” = -2).
- **NRC**: Categorizes emotions (anger, joy, etc.).

## Example: Hotel Sentiment Scores

| Hotel | Agoda Sentiment | Agoda Rating | Booking.com Sentiment | Booking.com Rating |
|-------|----------------|--------------|----------------------|-------------------|
| One World | 6.85 | 8.5 | 6.59 | 8.5 |
| Summer Suite | 7.27 | 8.4 | 7.1 | 8.7 |

These scores reflect overall sentiment from reviews, often aligning with ratings but providing deeper emotional insights.

## Conclusion

Sentiment analysis offers a powerful way to understand emotions in text, though it requires careful interpretation due to linguistic complexities. Using R’s `tm` and `syuzhet` packages, you can preprocess text, score sentiments, and visualize emotions, making it ideal for social media or review analysis.