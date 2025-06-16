---
layout: post
title: "文本聚类分析：基于相似性的文档分组"
date: 2025-06-16
tags:
  - 文本聚类分析
  - R
description: "使用R语言进行文本聚类分析：基于相似性的文档分组"
comments: true
---

大家周一快乐！最近世界局势动荡，中东冲突不断。这种混乱可能会影响我们对世界的认知。就像法国人说的 "C'est la vie"（这就是生活）。但无论未来如何，请记住瑞士人常说的 "La vie est belle"（生活是美好的）。

文本聚类分析通过内容相似性将文档分组，实现在R语言中自动对大型文本集合进行分类。

## 什么是文本聚类分析？

聚类分析将文档分组，使得同一组内的文档彼此之间的相似度高于与其他组中文档的相似度。相似性通常使用距离度量（如欧氏距离或余弦距离）来衡量。这对于无需人工干预地组织大型文档集合至关重要。

## 文本聚类的处理流程

文本聚类包含五个阶段：
1. **文本预处理**：通过去除标点符号、数字、停用词、多余空格并将文本转换为小写来清理原始文本。应用词干提取将单词还原为其基本形式。
2. **词项-文档矩阵(TDM)**：将文档表示为词频向量。
3. **TF-IDF标准化**：通过词频-逆文档频率(TF-IDF)对词项进行加权，以突出独特词项并减少常见词项的影响。
4. **距离计算与聚类**：计算文档向量之间的距离，并应用聚类算法（如K-means、层次聚类或HDBSCAN）。
5. **自动标注**：为每个聚类生成代表性标签（聚类中心）。

## 聚类算法

### K-均值聚类
K-means通过以下方式将文档划分为K个簇：
- 选择K个种子文档。
- 根据欧氏距离将每个文档分配给最近的种子。
- 迭代更新聚类中心。

**示例：K-均值聚类**

```R
library(tm)
library(proxy)

# 文本预处理
# 假设"TextFile"目录下有一些文本文档
mytext <- DirSource("TextFile")
docs <- VCorpus(mytext)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stripWhitespace)

# 创建TDM并应用TF-IDF
tdm <- DocumentTermMatrix(docs)
tdm.tfidf <- weightTfIdf(tdm)
tdm.tfidf <- removeSparseTerms(tdm.tfidf, 0.999)
tfidf.matrix <- as.matrix(tdm.tfidf)

# 计算余弦距离
dist.matrix <- dist(tfidf.matrix, method = "cosine")

# K-means聚类
kmeans <- kmeans(tfidf.matrix, centers = 3, iter.max = 100)
```

这段代码对文本进行预处理，创建TF-IDF加权的TDM，计算余弦距离，并使用3个中心点应用K-means聚类。

### 层次聚类
层次聚类通过自底向上合并文档来构建树状图，基于相似性度量（如完全连接、平均连接、单连接或Ward方法）。

### HDBSCAN（层次密度空间聚类）
HDBSCAN在以下方面表现出色：
- 任意形状的聚类
- 不同大小和密度的簇
- 噪声和异常值处理
它识别数据中的密集区域，将簇与稀疏区域分开。

**与K-means的比较**：
- K-means假设簇为球形，对噪声敏感。
- HDBSCAN适应复杂的簇形状，有效处理噪声。

## 文本聚类 vs. 主题模型

| **方面**        | **文本聚类**                          | **主题模型**                          |
|-------------------|--------------------------------------|-------------------------------------|
| **目标**         | 将文档分组为连贯的类别               | 识别文档中的潜在主题                |
| **方法**         | 相似性度量（如基于距离）             | 概率分布（如LDA）                   |
| **输出**         | 文档被分配到一个簇                   | 与主题相关的词簇                    |

## 聚类可视化

使用主成分分析或肘部图来可视化聚类，确定最佳K值：

```R
library(colorspace)
points <- cmdscale(dist.matrix, k = 2)  # 降维到2D
plot(points, main = "K-均值聚类", col = as.factor(kmeans$cluster))

# 绘制肘部图确定最佳K值
cost_df <- data.frame()
for(i in 1:20) {
  kmeans <- kmeans(tfidf.matrix, centers = i, iter.max = 100)
  cost_df <- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}
names(cost_df) <- c("cluster", "cost")
plot(cost_df$cluster, cost_df$cost, type = "b")
```

肘部图通过显示增加聚类数量带来的成本降低趋势，帮助确定最佳聚类数量。

让我们通过一个真实场景将所有内容整合起来。

## 真实数据集示例：20个新闻组

在深入理论之前，让我们通过20个新闻组数据集的实践示例来探索。该数据集包含约20,000个新闻组文档，分为20个不同的新闻组类别。这个数据集通常用于文本分类和聚类任务。

```r
# 安装所需包（如果尚未安装）
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(tm)) install.packages("tm")
if (!require(wordcloud)) install.packages("wordcloud")
if (!require(factoextra)) install.packages("factoextra")

# 加载所需库
library(tidyverse)
library(tm)
library(wordcloud)
library(factoextra)

# 创建示例中文新闻数据集（英文翻译）
set.seed(123)
news_data <- data.frame(
  text = c(
    # 科技
    "人工智能技术正在快速发展，改变着我们的生活方式。",
    "5G网络的普及将推动物联网技术的广泛应用。",
    "量子计算研究取得重大突破，计算能力大幅提升。",
    "区块链技术在金融领域具有广阔的应用前景。",
    
    # 体育
    "中国女排在奥运会上再次夺冠，为国家争光。",
    "NBA季后赛激战正酣，各支球队竞争激烈。",
    "世界杯预选赛即将开打，各国家队积极备战。",
    "中超联赛新赛季即将拉开帷幕。",
    
    # 财经
    "A股市场今日震荡上行，科技板块表现强势。",
    "央行出台新政策，支持实体经济发展。",
    "人民币汇率保持基本稳定，外汇储备充足。",
    "中国经济增长平稳，消费市场活力增强。",
    
    # 教育
    "教育部出台新政策，推进素质教育改革。",
    "高校招生工作启动，考生填报志愿需谨慎。",
    "在线教育平台快速发展，改变传统学习方式。",
    "新修订的职业教育法通过，职业教育迎来新发展。"
  ),
  category = rep(c("科技", "体育", "财经", "教育"), each = 4),
  stringsAsFactors = FALSE
)

# 自定义停用词
my_stopwords <- c("the", "and", "is", "in", "to", "of", "are", "with", "for", "as", "has", "have")

# 文本预处理函数
preprocess_text <- function(texts) {
  # 转换为小写
  texts <- tolower(texts)
  # 去除标点
  texts <- gsub("[[:punct:]]", "", texts)
  # 去除数字
  texts <- gsub("[0-9]+", "", texts)
  # 去除多余空格
  texts <- gsub("\\s+", " ", trimws(texts))
  return(texts)
}

# 预处理文本
processed_texts <- preprocess_text(news_data$text)

# 创建语料库
corpus <- VCorpus(VectorSource(processed_texts))

# 进一步处理文本
processed_corpus <- corpus %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, my_stopwords) %>%
  tm_map(stripWhitespace)

# 创建文档-词项矩阵
dtm <- DocumentTermMatrix(processed_corpus, control = list(
  wordLengths = c(3, Inf),  # 最小词长为3
  bounds = list(global = c(1, Inf))  # 词项至少出现在1个文档中
))

# 转换为矩阵并应用TF-IDF
tfidf <- weightTfIdf(dtm)
tfidf_matrix <- as.matrix(tfidf)

# 执行K-means聚类（4个簇对应4个类别）
set.seed(123)
k <- 4
kmeans_result <- kmeans(tfidf_matrix, centers = k, nstart = 25)

# 将聚类结果添加到原始数据
news_data$cluster <- kmeans_result$cluster

# 评估聚类结果与真实类别的匹配程度
confusion_matrix <- table(Predicted = kmeans_result$cluster, 
                         Actual = as.factor(news_data$category))
print("混淆矩阵:")
print(confusion_matrix)

# 使用PCA可视化聚类结果
pca_result <- prcomp(tfidf_matrix, scale. = FALSE)
plot_data <- data.frame(
  PC1 = pca_result$x[,1], 
  PC2 = pca_result$x[,2],
  Cluster = as.factor(kmeans_result$cluster),
  Category = news_data$category
)

# 绘制改进后的可视化图
ggplot(plot_data, aes(x = PC1, y = PC2, color = Cluster, shape = Category)) +
  geom_point(size = 4, alpha = 0.8) +
  theme_minimal() +
  labs(title = "新闻文章K-means聚类",
       subtitle = "TF-IDF向量的PCA投影",
       x = "主成分1",
       y = "主成分2") +
  theme(legend.position = "bottom")

# 为每个聚类创建词云
par(mfrow = c(2, 2))
for (i in 1:k) {
  cluster_docs <- which(kmeans_result$cluster == i)
  cat("\n聚类", i, "包含", length(cluster_docs), "个文档\n")
  
  # 获取该聚类中的文档
  docs_in_cluster <- processed_corpus[cluster_docs]
  
  # 为聚类创建词云
  wordcloud(docs_in_cluster, 
           max.words = 30, 
           random.order = FALSE,
           colors = brewer.pal(8, "Dark2"),
           scale = c(3, 0.5),
           main = paste("聚类", i))
}
```
![text-clustering-word-cloud-zlu-me](/assets/images/uploads/text-clustering-word-cloud-zlu-me.png)

此示例演示了如何：
1. 从20个新闻组数据集加载和预处理文本数据
2. 创建带有TF-IDF加权的文档-词项矩阵
3. 对文本数据执行K-means聚类
4. 使用PCA可视化聚类结果
5. 创建词云来探索每个聚类的主题

混淆矩阵显示了聚类结果与实际新闻组类别的匹配程度，可视化将帮助您在降维空间中观察聚类的分离情况。

```R
[1] "混淆矩阵:"
         Actual
Predicted 教育 体育 科技 财经
        1    0   0   2   0
        2    0   0   1   0
        3    3   4   1   4
        4    1   0   0   0

聚类 1 包含 2 个文档
聚类 2 包含 1 个文档
聚类 3 包含 12 个文档
聚类 4 包含 1 个文档
```

## 结论

文本聚类分析通过预处理、TF-IDF和K-means或HDBSCAN等算法实现文档自动分类。它非常适合组织大型文本数据集，通过关注文档相似性而非主题内容来补充主题建模。使用R的`tm`和`proxy`包可以有效地实现和可视化聚类结果。
