---
layout: post
title: "情感分析：解码文本中的情感"
date: 2025-06-16
tags:
  - 情感分析
  - R
description: "情感分析：解码文本中的情感"
comments: true
---

基于之前关于文本聚类和文本模型的博客，我们现在可以深入探讨一个经典主题 - 情感分析。情感分析通过计算方式识别和分类文本中的情感，帮助理解公众意见或消费者反馈。

## 什么是情感分析？

情感分析确定文本背后的情感基调，将其分类为积极、消极或中性。它被广泛用于社交媒体监控和理解消费者需求。

### 为什么使用情感分析？
- **公众意见**：评估对话题或品牌的情绪。
- **消费者洞察**：快速识别客户反应（例如，Expedia加拿大的商业案例）。

### 挑战
人类语言很复杂，机器在处理讽刺等细微差别时存在困难（例如，"太～～～～好了！！"可能被误读为积极）。算法正在不断发展以处理这些情况，但还不能达到100%的准确性。

## 情感分析流程

1. **文本预处理**：
   - **分词**：将文本分割成单词或短语。
   - **停用词过滤**：删除常见词（如"和"、"的"）。
   - **否定处理**：处理否定词（如"不好"与"不是不好"）。
   - **词干提取**：将词还原为词根形式（如"跑步"到"跑"）。
2. **情感分类**：使用词典或算法分配极性（积极/消极）。
3. **情感评分**：量化情感强度，考虑大写等因素（如"GOOD"表示更强的情绪）。

### 示例数据
| 文本 | 情感 |
|------|----------|
| 喜欢悉尼的德国面包店... | 积极 |
| @VivaLaLauren 我的也坏了！... | 消极 |
| @Mofette 太棒了！愿原力与你同在... | 积极 |

## R语言中的情感分析

使用R的`tm`、`syuzhet`和其他包，我们可以预处理文本并分析情感。

### 预处理和词云

```R
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)

# 用于情感分析的示例文本数据
text <- c(
  "我绝对喜欢这个产品！它超出了我所有的期望，运行完美。",
  "服务太差了。我从未对一次购买如此失望过。",
  "这个还行。不是很好，但也不差。我想它能完成工作。",
  "客服团队非常乐于助人，几分钟内就解决了我的问题。太棒了！",
  "考虑到价格，质量相当差。根据评论我期望会更好。",
  "这是我今年买过的最好的东西。每一分钱都值得！",
  "我对发货延迟感到非常沮丧。产品很好，但等待时间让人无法接受。",
  "说明不够清晰，但一旦我弄明白了，产品就如描述的那样工作。",
  "我不会向任何人推荐这个。完全浪费金钱和时间。",
  "设计很漂亮，使用非常方便。我对这次购买非常满意！"
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

# 创建词-文档矩阵
dtm <- TermDocumentMatrix(docs)
dtm_m <- as.matrix(dtm)
dtm_v <- sort(rowSums(dtm_m), decreasing=TRUE)
dtm_d <- data.frame(word = names(dtm_v), freq=dtm_v)

# 生成词云，调整参数
# 设置图形边距（下、左、上、右）
par(mar = c(0, 0, 0, 0))  # 移除所有边距

# 创建新图形，设置更大尺寸
png("wordcloud.png", width = 10, height = 8, units = "in", res = 300)  # 高分辨率

set.seed(1234)
wordcloud(
  words = dtm_d$word, 
  freq = dtm_d$freq, 
  min.freq = 1,
  max.words = 50,
  random.order = FALSE, 
  rot.per = 0,            # 不旋转
  scale = c(4, 0.8),      # 最大和最小词之间的比例
  colors = brewer.pal(8, "Dark2"),
  vfont = c("sans serif", "plain"),
  use.r.layout = TRUE     # 更好的布局算法
)

dev.off()  # 关闭设备

# 显示保存的图像
if (requireNamespace("png", quietly = TRUE) && requireNamespace("grid", quietly = TRUE)) {
  library(png)
  library(grid)
  if (file.exists("wordcloud.png")) {
    img <- png::readPNG("wordcloud.png")
    grid::grid.raster(img)
  } else {
    warning("未找到词云图像。请检查文件路径。")
  }
} else {
  warning("请安装'png'和'grid'包以显示词云。")
}
```
![词云](/assets/images/uploads/sentiment-analysis-word-cloud-zlu-me.png)
这段代码预处理文本，去除噪音，并在词云中可视化频繁出现的词。

### 情感评分

使用`syuzhet`进行不同词典的情感分析：

```R
library(syuzhet)
library(ggplot2)

# 使用多种方法进行情感评分
syuzhet_vector <- get_sentiment(text, method="syuzhet")
bing_vector <- get_sentiment(text, method="bing")
afinn_vector <- get_sentiment(text, method="afinn")

# 比较前几个分数
rbind(
  sign(head(syuzhet_vector)),
  sign(head(bing_vector)),
  sign(head(afinn_vector))
)

# 使用NRC进行情感分类
d <- get_nrc_sentiment(text)
td <- data.frame(t(d))
td_new <- data.frame(rowSums(td))
names(td_new) <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)

# 创建更具信息量的图表
ggplot(td_new, aes(x = reorder(sentiment, count), y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +  # 移除图例，因为它是多余的
  labs(title = "情感分析结果",
       x = "情感",
       y = "计数") +
  scale_fill_brewer(palette = "Set3") +
  coord_flip()  # 翻转坐标以获得更好的可读性

# 创建多个可视化
# 1. 基本情感分数比较
sentiment_scores <- data.frame(
  Text = 1:length(text),
  Syuzhet = syuzhet_vector,
  Bing = bing_vector,
  Afinn = afinn_vector
)

# 重塑数据以便绘图
sentiment_long <- tidyr::pivot_longer(sentiment_scores, 
                                    cols = c(Syuzhet, Bing, Afinn),
                                    names_to = "Method",
                                    values_to = "Score")

# 图表1：比较不同的情感评分方法
p1 <- ggplot(sentiment_long, aes(x = Text, y = Score, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "情感评分方法比较",
       x = "文本样本",
       y = "情感分数") +
  scale_fill_brewer(palette = "Set2")

# 图表2：NRC情感分析（上面已创建）
p2 <- ggplot(td_new, aes(x = reorder(sentiment, count), y = count, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  labs(title = "情感分析结果",
       x = "情感",
       y = "计数") +
  scale_fill_brewer(palette = "Set3") +
  coord_flip()

# 图表3：词云（上面已创建）
# 词云已保存为"wordcloud.png"

# 显示所有图表
print(p1)
print(p2)

# 打印汇总统计
cat("\n情感分数汇总：\n")
print(summary(sentiment_scores[, -1]))

# 打印最积极和最消极的文本
cat("\n最积极的文本：\n")
print(text[which.max(syuzhet_vector)])
cat("\n最消极的文本：\n")
print(text[which.min(syuzhet_vector)])
```

![情感评分](/assets/images/uploads/sentiment-score-zlu-me.png)


![情感评分](/assets/images/uploads/emotion-analysis-zlu-me.png)



```R

情感分数汇总：
    Syuzhet            Bing           Afinn      
 Min.   :-1.750   Min.   :-2.00   Min.   :-5.00  
 1st Qu.:-0.250   1st Qu.: 0.00   1st Qu.:-0.75  
 Median : 0.325   Median : 0.00   Median : 1.50  
 Mean   : 0.600   Mean   : 0.80   Mean   : 2.20  
 3rd Qu.: 1.738   3rd Qu.: 2.75   3rd Qu.: 5.75  
 Max.   : 3.150   Max.   : 4.00   Max.   :10.00  

最积极的文本：
[1] "客服团队非常乐于助人，几分钟内就解决了我的问题。太棒了！"

最消极的文本：
[1] "我对发货延迟感到非常沮丧。产品很好，但等待时间让人无法接受。"
```



这段代码使用`syuzhet`、`bing`和`afinn`词典进行情感评分，并使用NRC词典可视化情感（如喜悦、悲伤）。

## 基于词典的分析

像`bing`和`afinn`这样的词典为词分配情感分数：
- **Bing**：二元（积极/消极，例如"放弃"=消极）。
- **Afinn**：数值分数（例如"放弃"=-2）。
- **NRC**：对情感进行分类（愤怒、喜悦等）。

## 示例：酒店情感分数

| 酒店 | Agoda情感 | Agoda评分 | Booking.com情感 | Booking.com评分 |
|-------|----------------|--------------|----------------------|-------------------|
| One World | 6.85 | 8.5 | 6.59 | 8.5 |
| Summer Suite | 7.27 | 8.4 | 7.1 | 8.7 |

这些分数反映了评论的整体情感，通常与评分一致，但提供了更深层次的情感洞察。

## 结论

情感分析提供了一种强大的方式来理解文本中的情感，尽管由于语言的复杂性需要谨慎解释。使用R的`tm`和`syuzhet`包，你可以预处理文本、评分情感并可视化情绪，使其成为社交媒体或评论分析的理想工具。 