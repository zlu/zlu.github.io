---
layout: post
title: Introduction to Vector Database
date: 2025-06-09 06:33 +0800
tags:
  - vector database
  - database
description: Introduction to vector database
comments: true
---

A vector database is a database that stores and manages vector data.  Vector data is data that is represented as a vector, such as a point in space or a vector in time.  Vector databases are used in a variety of applications, such as image and video search, natural language processing, and recommendation systems.  In machine learning, we typically use vector db to storage embedding text data from models like BERT or OpenAI; image data (embeddings from CNNs or CLIP), and audio/video/genomic data.  Instead of traditional exact match queries like SQL's WHERE clause, vector db supports *similarity search* like "find the top 10 documents most similar to the self-attention paper".  Vector db are used in applications involving semantic search, recommendation systems, anomaly detection, and retrieval-augmented generation (RAG).

Milvus is an open-source vector db optimized for storing and retrieving embedding vectors.  It is a distributed vector db that can scale to handle large amounts of data.  It is built on top of Apache Arrow and uses a columnar storage format to store data.  It also provides gRPC as well as REST API for easy integration with other applications.  It is written in Go and is available for Linux, Windows, and macOS.  It capably handles large-scale (billions of vectors) similarity search using **Approximate Nearest Neighbor (ANN)** algorithms like HNSW, IVF, and ANNOY.

Milvus lite is a lightweight, local-only version of Milvus.  It runs either in-memory or on file system.  It is ideal for small-scale applications or for testing purposes.

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