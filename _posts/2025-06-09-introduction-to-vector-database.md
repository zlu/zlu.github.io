---
layout: post
title: Introduction to Vector Database
date: 2025-06-09 06:33 +0800
tags:
  - vector database
  - database
  - ANN
description: Introduction to vector database
comments: true
---

Update: 2025-06-19
- Add a new section on ANN algorithms.

A vector database is a database that stores and manages vector data. Vector data is data that is represented as a vector, such as a point in space or a vector in time. Vector databases are used in a variety of applications, such as image and video search, natural language processing, and recommendation systems. In machine learning, we typically use vector db to storage embedding text data from models like BERT or OpenAI; image data (embedding from CNNs or CLIP), and audio/video/genomic data. Instead of traditional exact match queries like SQL's WHERE clause, vector db supports _similarity search_ like "find the top 10 documents most similar to the self-attention paper". Vector db are used in applications involving semantic search, recommendation systems, anomaly detection, and retrieval-augmented generation (RAG).

The reason for the existence of vector db is that it preserves semantic meaning
of the text data, meaning similar content yields similar vector (cosine
similarity).

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("ChatGPT is somewhat intelligent.")
```

Models such as BERT generates vector embeddings for the text. We then create
index, which is a structure that allows **fast approximate nearest neighbor (ANN)**
search. ANN is the core technique behind modern vector db. It is actually a
family of similarity search algorithms.

#### Inverted File Index (IVF)

IVF clusters the dataset into _k_ coarse centroids via **k-means**. Each vector
is assigned to its nearest neighbor. At query time, it first searches only the top-n
centroids closest to the query (coarse quantization). It then perform an exact
search in those clusters. Given the efficient time complexity for such search is
Olog(k) + small local search, and the small memory consumption,  we generally
use it as a baseline.  

#### Graph-Based

There are several Graph-Based algorithms in this category.  Take **Hierarchical
Navigatable Small World Graphs (HNSW)** for example, which builds a multi-layer
graph, where the top layers are sparse with long-range links, and the bottom
layers are dense with local links.  During the querying phase, it performs
greedy search starting from the top and descending layer by layer.  The
navigation of the graph uses neighbor vectors with the shortest distance.  It is
an extremely fast algorithm with a time complexity of Olog(n), with added
benefit of high accuracy.

#### Hash-Based

One of the Hash-Based algorithm is called **Locality-Sensitive Hash (LSH)**.
The algorithm projects high-dimensional vectors into hash buckets such that
similar vectors map to the same buckets with high probability.  It also uses
multiple hash tables to increase accuracy.  Time complexity is then O(1) + small
candidate search with fast insert.  Although this algorithm suffers from poor
performance in very high dimensions.

Finally we should mention that there are several means for the similarity
calculation: L2 Euclidean, Cosine, inner product, and Jaccard/Hamming.

Here is an example of IVF + PQ:

```python
 
import faiss
import numpy as np

d = 128  # dimension
nb = 100000  # number of vectors
nlist = 100  # number of partitions

# generate training data
xb = np.random.random((nb, d)).astype('float32')

# index with IVF + PQ
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8)  # 16 subquantizers, 8 bits each
index.train(xb)
index.add(xb)

# query
xq = np.random.random((1, d)).astype('float32')
index.nprobe = 10  # how many clusters to search
D, I = index.search(xq, 5)  # top 5 results
```


Milvus is an open-source vector db optimized for storing and retrieving embedding vectors. It is a distributed vector db that can scale to handle large amounts of data. It is built on top of Apache Arrow and uses a columnar storage format to store data. It also provides gRPC as well as REST API for easy integration with other applications. It is written in Go and is available for Linux, Windows, and macOS. It capably handles large-scale (billions of vectors) similarity search using **Approximate Nearest Neighbor (ANN)** algorithms like HNSW, IVF, and ANNOY.

Milvus lite is a lightweight, local-only version of Milvus. It runs either in-memory or on file system. It is ideal for small-scale applications or for testing purposes.

Typical operations are the following:

#### Connection

```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
```

#### Define Schema and Create Collection

```python
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="My vector collection")
collection = Collection(name="my_collection", schema=schema)
```

#### Insert Vector Data

```python
import numpy as np

ids = [1, 2, 3]
vectors = np.random.rand(3, 128).tolist()

collection.insert([ids, vectors])
```

#### Create Index

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

#### Load and Search

```python
collection.load()

search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
query_vector = [np.random.rand(128).tolist()]
results = collection.search(query_vector, "embedding", search_params, limit=2)

for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance}")
```

#### Filtering with Hybrid Search

Milvus allows combining vector search with structured filters (e.g., WHERE id > 10). You can include a filter in the search() method if schema supports it.

In a future blog, we will discuss a specific use case of vector db's application in RAG (Deepsearcher).
