---
layout: post
title: The Glorious Graphic Convolutional Network - From Theory to Practice
date: 2025-06-07 20:43 +0800
tags:
  - python
  - artificial intelligence
  - machine learning
  - graph convolutional network
  - GCN
description: The Glorious Graphic Convolutional Network - From Theory to Practice
---

Graph Convolutional Networks (GCNs) have revolutionized the field of graph-based machine learning, enabling deep learning on non-Euclidean structures such as social networks, citation graphs, and molecular structures. In this blog, we will explain the intuition, math principles, and provide code snippets to help you understand and implement a basic GCN.  Traditional CNNs work well with grid-like data such as images or text but struggle with arbitrary graphs where node connections vary and there is no fixed spatial locality. GCNs overcome this by performing convolution operations on graphs.

## Graph Notation Primer

Let a graph be defined as G = (V, E), where:
- $V$: set of nodes
- $E$: set of edges
- $A \in \mathbb{R}^{N \times N}$: adjacency matrix
- $X \in \mathbb{R}^{N \times F}$: node feature matrix

Here, $N$ is the number of nodes and $F$ is the number of input features per node.

### Ajacency Matrix
Adjacency Matrix is a way to represent the connections (edges) between nodes in a graph.
- For a graph with $N$ nodes, $A$ is an $N \times N$ matrix.
- If there is an edge between node $i$ and node $j$, then $A_{ij} = 1$ (or the edge weight, if weighted); otherwise, $A_{ij} = 0$.
- In undirected graphs, $A$ is symmetric ($A_{ij} = A_{ji}$).
- Example for a 3-node graph where node 0 is connected to node 1 and 2:
  $$
  A = \begin{bmatrix}
  0 & 1 & 1 \\
  1 & 0 & 0 \\
  1 & 0 & 0
  \end{bmatrix}
  $$

![adjacency matrix](/assets/images/uploads/adjacency-matrix.png)

### Node feature matrix

Node Feature Matrix stores the features (attributes) of each node in the graph.
- $N$ is the number of nodes, $F$ is the number of features per node.
- Each row $X_i$ is a feature vector for node $i$.
- For example, if each node has 3 features (say, age, income, and group), and there are 4 nodes:
  $$
  X = \begin{bmatrix}
  23 & 50000 & 1 \\\\
  35 & 60000 & 2 \\\\
  29 & 52000 & 1 \\\\
  41 & 58000 & 3
  \end{bmatrix}
  $$
- These features are what the GCN uses as input to learn from the graph.

Together, they are essential inputs for a Graph Convolutional Network:
- Adjacency Matrix $A$ describes how nodes are connected.  
- Node Feature Matrix $X$ describes what each node “looks like” (its features).  

### GCN Layer Formula (Kipf & Welling, 2016)

The core GCN layer is expressed as such:

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

This formula packs a lot of information and we will carefully unpack it below:

#### Inputs:

- $H^{(l)}$: Node features from the previous layer (for the first layer, $H^{(0)} = X$, the input features)
- $\tilde{A} = A + I$: The adjacency matrix with self-loops added (I is the identity matrix).  Self-loop in a graph are edges that connect a node to itself.  In the context of adjacency matrix, a self-loop at node $i$ is represented by $\tilde{A}_{ii} = 1$.  When we add self-loops, we create a new matrix: $\tilde{A} = A + I$.  This step is important because we want to preserve a node's own features in the aggregation.  Otherwise, it would only get information from its neighbors, losing its own characteristics.
- $\tilde{D}$: Diagonal degree matrix of $\tilde{A}$ (contains the number of connections for each node, including self-loops)
- $W^{(l)}$: Trainable weight matrix for layer l
- $\sigma$: Non-linear activation function (like ReLU)

#### Key Operations: 

- Message Passing: 
  - $\tilde{A}H^{(l)}$: Each node aggregates feature vectors from its neighbors
  - Adding self-loops ($\tilde{A} = A + I$) ensures nodes include their own features in aggregation.
- Normalization: Prevents the scale of features from changing too much across layers
  and helps with training stability by normalizing by node degrees.
  - $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$: This step is called symmetric normalization or renormalization trick.  Without normalization, nodes with many connections (high degree) would have larger feature values after aggregation, which can cause numerical instability and difficulties in training.
  - $\tilde{D}$: Degree matrix (diagonal matrix where $\tilde{D}{ii} = \sum_j \tilde{A}{ij}$)
  - $\tilde{D}^{-1/2}$: The inverse square root of the degree matrix
  - Left Multiplication ($\tilde{D}^{-1/2} \tilde{A}$): Divides each row by the square root of the node's degree.  This normalizes the influence of each node's outgoing messages.
  - Right Multiplication ($\cdot \tilde{D}^{-1/2}$): Divides each column by the square root of the node's degree.  This normalizes the influence of each node's incoming messages.
  - Consider a simple graph with 3 nodes:

  ```bash
  Node 0 is connected to Node 1
  Node 1 is connected to Nodes 0 and 2
  Node 2 is connected to Node 1
  ```

  With self-loops:

  ```bash
  A = [[1, 1, 0],
    [1, 1, 1],
    [0, 1, 1]]
    
  D = [[2, 0, 0],
      [0, 3, 0],
      [0, 0, 2]]  # degrees: 2, 3, 2

  D^(-1/2) = [[1/√2, 0,     0   ],
              [0,    1/√3,  0   ],
              [0,    0,     1/√2]]
  ```
  
  The normalized matrix is:

  ```bash
  D^(-1/2)AD^(-1/2) = 
    [[1/2,   1/√6,    0   ],
    [1/√6,  1/3,    1/√6  ],
    [0,     1/√6,    1/2  ]]
  ```

At each layer, a node aggregates information from its neighbors (including itself). The deeper the network, the farther the information propagates.  Each node's new representation is a weighted average of its own features and its neighbors' features.  The weights are learned through the training process.  The normalization ensures that nodes with many neighbors don't dominate the learning.  In a social network, where each person (node) has some features (age, interests, etc.), the GCN layer lets each person update their understanding based on their friends' information.  The normalization ensures that popular people (with many friends) don't overwhelm the learning process.

### GCN implementation for node classification on Cora dataset

The Cora dataset is a citation network where nodes represent academic papers and edges represent citations.  Each paper has a set of features (e.g., author, title, abstract) and a label (e.g., subject of the paper).  There are altogether 2,780 papers (nodes) and 5429 citations (edges).  Each paper is represented by a binary word vector indicating the presence (1) or absence (0) of 1,433 unique words from a dictionary.  Papers are classified into 7 categories (e.g. Neural Networks, Probabilistic Methods,etc.)  The goal is to predict the category of each paper based on its features and citation relationships.

#### Model Architecture

The GCN model has 2 layers:

```python
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)  # Input to hidden
        self.conv2 = GCNConv(16, dataset.num_classes)        # Hidden to output
```

The first GCN layer takes the input features (1,433 dimensions) and reduces them to 16 dimensions.  The second GCN layer takes the 16 dimensions and reduces them to 7 dimensions.


#### Forward function

```python
def forward(self):
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)  # First GCN layer
    x = F.relu(x)                  # Non-linearity
    x = F.dropout(x, training=self.training)  # Optional dropout
    x = self.conv2(x, edge_index)  # Second GCN layer
    return F.log_softmax(x, dim=1)  # Log probabilities for each class
```

`x = self.conv1(x, edge_index)` does a few things.  It adds self-loops to the graph, computes the normailized adjacency matrix  $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$, multiplies with input features the weights $H^{(l)} W^{(l)}$, and applies the normalization and aggregation.  Basically all the hard math is taken care of by the GCNConv layer :)  `F.relu(x)` applies the ReLU activation function, and `F.dropout(x, training=self.training)` applies dropout to prevent overfitting.  The second GCN layer `x = self.conv2(x, edge_index)` does the same thing, but with different weights $H^{(l)} W^{(l)}$.

#### Training Process

```python
model = GCN()
data = dataset[0]  # Get the first graph object
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

We use Adam optimizer with weight decay.  Adam is an adaptive learning rate optimization algorithm.  It combines the benefits of two other optimizers: AdaGrad and RMSProp.  It maintains per-parameter learning rates and uses moving average of both the gradients and squared gradients.  As sparse gradients is common in GNNs, it makes sense to apply Adam.  It takes two main parameters. `lr` is the learning rate, and `weight_decay` is the L2 regularization parameter.  Weight decay prevents overfitting by adding a penalty term to the loss function and pushes the model weigths towards smaller values, preventing any single weight from becoming too large.  With L2, the original loss $L(\theta)$ becomes $L(\theta) + \lambda \sum \theta_i^2$ where $\lambda$ is the weight decay parameter.  weight_decay=5e-4 means $\lambda = 0.0005$.  It prevents overfitting by keeping weights small and makes the model more generalizable to unseen data.`loss = F.nll_loss(...)` is a Negative Log-Likelihood Loss (NLL).  it is commonly used for classification tasks.  It measures how well the model's predicted probabilities match the true abels.  For a single sample, it is expressed as $-\log(p_{\text{true class}})$.  If the model is 100% confident in the correct class, then the loss is 0.  If the model 50% confident, then the loss is 0.  `data.train_mask` is a boolean mask indicating which nodes are in the training set.  `data.y` is the label of each node.  We only use nodes with train_mask is True for training.  `val_mask` are nodes used for validation and `test_mask` are nodes used for final evaluation.  Like many graph datasets, where labels are available to only a small subset of nodes, model learns from both labeled (via supervised loss) and unlabled nodes (via graph structure).  Thus it is semi-supervised learning.  with a total of 2,708 nodes in Cora dataset, about 140 nodes (5%) are used for training, 500 for validation, and 1000 for testing.  The GCN assumes that connected nodes are likely to be similar.  This is cally Homophily Assumption, which is built into the learning algorithm.  The GCN's message passing directly encodes these biases.

#### Model Evaluation

```python
model.eval()
pred = model().argmax(dim=1)  # Get predicted classes
correct = pred[data.test_mask] == data.y[data.test_mask]
accuracy = int(correct.sum()) / int(data.test_mask.sum())
```

Now putting everything together.  But first, install the necessary package:

`pip install torch-geometric`

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load data
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training loop
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluation

model.eval()
pred = model().argmax(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
accuracy = int(correct.sum()) / int(data.test_mask.sum())
print(f'Test Accuracy: {accuracy:.4f}')
```
This is the result:

```
Epoch 0, Loss: 1.9515
Epoch 20, Loss: 0.1116
Epoch 40, Loss: 0.0147
Epoch 60, Loss: 0.0142
Epoch 80, Loss: 0.0166
Epoch 100, Loss: 0.0155
Epoch 120, Loss: 0.0137
Epoch 140, Loss: 0.0124
Epoch 160, Loss: 0.0114
Epoch 180, Loss: 0.0107
Test Accuracy: 0.8100
```

We can see that the model is able to achieve a quite decent accuracy (81%) without seeing many labeled nodes.  This demonstrates the power of the graph structure in combination with node features.  In the next blog, we'll take a look at EvolveGCN, which is a dynamic GCN model that can handle dynamic graph data. 