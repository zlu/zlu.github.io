---
layout: post
title: "Unstructured Text Mining in R: A Beginner's Guide"
date: 2025-06-14
tags:
  - data preparation
  - machine learning
  - data visualization
  - feature scaling
description: "Unstructured Text Mining in R: A Beginner's Guide"
comments: true
---

Text mining, also known as text analytics, extracts meaningful insights from unstructured text data, which constitutes about 80% of global data. This blog explores key concepts and practical R-based techniques for text mining and web scraping.

## What is Text Mining?

Text mining uses computational techniques to derive structured information from unstructured text sources like books, reports, articles, blogs, and social media posts. It automates the discovery of knowledge from vast datasets, enabling summarization and analysis.

### Key Points:
- **Unstructured Text**: Data in free-form formats (e.g., emails, social media, documents).
- **Goal**: Extract high-quality, structured information for analysis.
- **Applications**: Sentiment analysis, topic modeling, and information retrieval.

## Extracting Text in R with the `tm` Package

The `tm` (text mining) package in R is a powerful tool for text mining, using a *corpus*—a collection of text documents—as its core data structure. A corpus allows batch processing of multiple documents.

### Types of Corpus:
- **VCorpus (Volatile Corpus)**: Temporary, stored in memory, deleted when the R session ends.
- **PCorpus (Permanent Corpus)**: Stored externally, persists across sessions.

### Predefined Sources:
- **DirSource**: Reads text from a directory.
- **VectorSource**: Handles text from vectors.
- **DataframeSource**: Processes data frame-like structures.

### Example: Creating a Corpus with VectorSource

This code creates a volatile corpus from a vector of text strings and inspects its contents.

```R
library(tm)
texts <- c("Hi!", "Welcome to My Blog!", "Blog1, 2, 3.....")
mytext <- VectorSource(texts)
mycorpus <- VCorpus(mytext)
inspect(mycorpus)
as.character(mycorpus[[1]])
```
Where:
- inspect(mycorpus) prints out the structure and metadata of the VCorpus object.  In this example, it shows that the VCorpus contains 3 documents, each of which is a `PlainTextDocument` of length 3, 19, and 16, respectively.
- as.character(mycorpus[[2]]) converts the second document in the corpus to a character vector.  In this example, it returns "Welcome to My Blog!".  The `[[]]` is used to access elements in Rlists, and since a corpus is essentially a list of documents, this is how you would access individual

```R
<<VCorpus>>
Metadata:  corpus specific: 0, document level (indexed): 0
Content:  documents: 3

[[1]]
<<PlainTextDocument>>
Metadata:  7
Content:  chars: 3

[[2]]
<<PlainTextDocument>>
Metadata:  7
Content:  chars: 19

[[3]]
<<PlainTextDocument>>
Metadata:  7
Content:  chars: 16
```


## Web Scraping for Text Data

Web scraping retrieves data from websites, often requiring parsing to extract relevant content from HTML. Tools like `readLines()`, `httr`, `XML`, and `rvest` simplify this process.

### Challenges:
- Web data is often embedded in complex HTML structures.
- Parsing is needed to isolate useful text.

### Techniques and Tools:
- **readLines()**: Reads raw text from a URL.
- **httr::GET()**: Fetches web content programmatically.
- **XML::htmlParse()**: Parses HTML to extract specific elements using XPath.
- **rvest::read_html()**: Reads and parses HTML, using CSS selectors for targeted scraping.

### Example: Web Scraping with `rvest`

rvest is an R package designed for web scraping, making it easy to extract data from HTML and XML web pages. It's part of the tidyverse ecosystem and is particularly user-friendly for those familiar with R's tidyverse syntax.

Key Functions in rvest:
- read_html(): Reads and parses HTML content from a URL or character string
  - Example: page <- read_html("https://example.com")
- html_nodes(): Extracts HTML elements using CSS selectors
  - Example: titles <- html_nodes(page, "h1")
- html_text(): Extracts text content from HTML nodes
  - Example: text_content <- html_text(titles)
- html_attr(): Extracts attributes from HTML elements (like href, src, etc.)
  - Example: links <- html_attr(links, "href")

This code scrapes specific elements from a webpage using html selectors.

```R
library(rvest)
url <- "https://zlu.me/teach"
page <- read_html(url)
nodes <- html_nodes(page, "h2")
texts <- html_text(nodes)
print(texts)
```

```R
[1] ""          "Teach@zlu" "留学辅导" 
```

### Scraping Structured Data

Here's an example that extracts course categories from a teaching website:

```R
library(rvest)
library(purrr)

# Scrape course categories from the teach page
url <- "https://zlu.me/teach"
page <- read_html(url)

# Extract all section headers (h3 elements)
headers <- html_nodes(page, 'h3') %>% 
  html_text() %>%
  keep(~nchar(.) > 0)  # Remove empty strings

# Print the headers
cat("Sections found on the page:\n")
walk(headers, ~cat("- ", ., "\n"))

# Extract course categories
categories <- html_nodes(page, 'h4') %>% 
  html_text() %>%
  keep(~nchar(.) > 0)  # Remove empty strings

# Print the course categories
cat("\nCourse Categories:\n")
walk(categories, ~cat("- ", ., "\n"))
```

And the partial results:

```R
Sections found on the page:
> walk(headers, ~cat("- ", ., "\n"))
-  Recent Posts 
-  About Me 
-  Popular Courses 
-  Student Success 
-  University Courses 
-  FAQs 
-  Book a Session 
-  Introduction 
-  About Me 
-  Popular Courses 
-  Course Categories 
-  Student Testimonials 
-  Frequently Asked Questions 
-  简介 
-  详细介绍 
-  Popular Courses 
-  课程分类 
-  学生评价或成功案例 
-  常见问题解答（FAQ） 
> 
> # Extract course categories
> categories <- html_nodes(page, 'h4') %>% 
+   html_text() %>%
+   keep(~nchar(.) > 0)  # Remove empty strings
> 
> # Print the course categories
> cat("\nCourse Categories:\n")

Course Categories:
> walk(categories, ~cat("- ", ., "\n"))
-  Machine Learning 
-  Artificial Intelligence 
-  Data Analysis 
-  Databases 
-  Python Programming 
-  CS Core 
-  Advanced Topics 
-  Machine Learning 
-  Artificial Intelligence 
-  Data Analysis 
-  Databases 
-  Python Programming 
-  CS Core 
-  Advanced Topics 
```


## Conclusion

Text mining and web scraping unlock insights from unstructured data. The `tm` package in R simplifies text extraction, while tools like `rvest` and `httr` enable efficient web scraping. By combining these techniques, you can process and analyze vast amounts of text data effectively.

Happy Saturday!