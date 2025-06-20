---
layout: post
title: '向量数据库简介'
date: 2025-06-09 14:46:00 +0800
tags:
  - 向量数据库
  - 数据库
  - 近似最近邻
  - ANN
description: 向量数据库简介
comments: true
---

更新 2025-06-20
- 添加关于 ChromaDB/SQLLite 采样的章节。
更新：2025-06-19
- 新增近似最近邻（ANN）算法相关内容。

向量数据库是一种存储和管理向量数据的数据库。向量数据是表示为向量的数据，例如空间中的点或时间序列中的向量。向量数据库在各种应用中使用，如图像和视频搜索、自然语言处理和推荐系统。在机器学习中，我们通常使用向量数据库来存储来自BERT或OpenAI等模型的嵌入文本数据；图像数据（来自CNN或CLIP的嵌入）以及音频/视频/基因组数据。与SQL的WHERE子句等传统精确匹配查询不同，向量数据库支持*相似性搜索*，例如"查找与自注意力论文最相似的10篇文档"。向量数据库用于涉及语义搜索、推荐系统、异常检测和检索增强生成（RAG）的应用。

向量数据库存在的意义在于它能够保留文本数据的语义信息，也就是说内容相似的数据会生成相似的向量（如余弦相似度）。

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("ChatGPT is somewhat intelligent.")
```

像BERT这样的模型会为文本生成向量嵌入。我们随后会建立索引（index），以支持**快速近似最近邻（ANN）**搜索。ANN是现代向量数据库的核心技术，实际上是一类相似性搜索算法的统称。

#### 近似最近邻（ANN）算法简介

- **倒排文件索引（IVF）**：IVF通过**k-means**将数据集聚类为k个粗粒度中心点（centroid），每个向量分配到最近的中心点。查询时，先只在与查询向量最接近的前n个中心点对应的簇中进行精确搜索（粗量化）。这种方法时间复杂度为Olog(k) + 局部小范围搜索，内存消耗小，常作为基线方法。

- **图结构算法**：如**分层可导航小世界图（HNSW）**，构建多层图结构，上层稀疏有长距离连接，下层稠密有局部连接。查询时自顶向下贪心搜索，每层选择距离最近的邻居，逐层下沉。该算法极快，时间复杂度Olog(n)，且准确率高。

- **哈希算法**：如**局部敏感哈希（LSH）**，将高维向量投影到哈希桶中，相似向量以高概率落入同一桶。通常用多个哈希表提升准确率。时间复杂度O(1) + 小规模候选集精查，插入速度快，但在高维空间表现不佳。

常见的相似度计算方式包括L2欧氏距离、余弦相似度、内积、Jaccard/Hamming等。

以下是IVF+PQ的示例：

```python
import faiss
import numpy as np

d = 128  # 向量维度
nb = 100000  # 向量数量
nlist = 100  # 分区数

# 生成训练数据
xb = np.random.random((nb, d)).astype('float32')

# 使用IVF+PQ建立索引
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8)  # 16个子量化器，每个8位
index.train(xb)
index.add(xb)

# 查询
xq = np.random.random((1, d)).astype('float32')
index.nprobe = 10  # 查询时搜索的簇数
D, I = index.search(xq, 5)  # 返回前5个结果
```
### 向量数据库

**ChromaDB** 使用 SQLite 作为其后端，并创建多个表来管理集合、嵌入、元数据等。以下是对 ChromaDB SQLite 数据库中常见表的简要说明：

- collections：存储每个集合（一组相关的嵌入/文档）的信息。
- collection_metadata：存储与集合相关的元数据（键值对）。
- embeddings：包含实际的向量嵌入及其关联文档的引用。
- embedding_metadata：存储每个嵌入的元数据（如文档 ID、标签等）。
- segments：用于管理数据分段，有助于高效存储和检索。
- segment_metadata：每个分段的元数据。
- embeddings_queue 和 embeddings_queue_config：用于管理队列中的嵌入操作，可能用于异步处理。
- databases：存储多租户环境下不同逻辑数据库的信息。
- maintenance_log：记录维护操作或后台任务。
- migrations：跟踪模式迁移（数据库版本控制）。
- embedding_fulltext_search 和相关表：支持对嵌入或文档进行全文搜索功能。

使用 SQLite CLI，我们可以方便地查看实际的 chromaDB：

命令 .schema TABLENAME 显示表的结构。在此示例中，
我们查看 embeddings 表。然后我们启用列标题，以便
SELECT 语句显示列标题和所选值的
对应值。

```sql

sqlite> .schema embeddings
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    segment_id TEXT NOT NULL,
    embedding_id TEXT NOT NULL,
    seq_id BLOB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (segment_id, embedding_id)
);

sqlite> .headers on
sqlite> .mode column
sqlite> select * from embeddings limit 1;
id  segment_id                            embedding_id                          seq_id  created_at
--  ----------------------------------- -  ------------------------------------  ------  -------------------
1   33e29416-2f64-4d96-a0db-4d95ea626ee6  5d7365f3-286c-45d7-be70-0e0d24df12b3  1       2025-06-13 12:49:01
```


**Milvus**是一个开源的向量数据库，专为存储和检索嵌入向量而优化。它是一个分布式向量数据库，可以扩展以处理大量数据。它构建在Apache Arrow之上，并使用列式存储格式存储数据。它还提供gRPC和REST API，便于与其他应用程序集成。它用Go编写，可在Linux、Windows和macOS上使用。它能够使用**近似最近邻（ANN）**算法（如HNSW、IVF和ANNOY）处理大规模（数十亿向量）的相似性搜索。

Milvus lite是Milvus的轻量级、仅本地版本。它可以在内存中或文件系统上运行。它非常适合小型应用程序或测试目的。

### 典型操作

#### 连接
```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
```

#### 定义模式和创建集合
```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="我的向量集合")
collection = Collection(name="my_collection", schema=schema)
```

#### 插入向量数据
```python
import numpy as np

ids = [1, 2, 3]
vectors = np.random.rand(3, 128).tolist()

collection.insert([ids, vectors])
```

#### 创建索引
```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

#### 加载和搜索
```python
collection.load()

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
query_vector = [np.random.rand(128).tolist()]
results = collection.search(query_vector, "embedding", search_params, limit=2)

for hit in results[0]:
    print(f"ID: {hit.id}, 距离: {hit.distance}")
```

#### 混合搜索过滤
Milvus允许将向量搜索与结构化过滤器（例如WHERE id > 10）结合使用。如果模式支持，您可以在search()方法中包含一个过滤器。

在未来的博客中，我们将讨论向量数据库在RAG（Deepsearcher）应用中的具体用例。
