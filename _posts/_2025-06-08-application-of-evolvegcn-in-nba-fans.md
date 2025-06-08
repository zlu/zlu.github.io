---
layout: post
title: "Analyzing NBA Fan Loyalty with EvolveGCN"
date: 2025-06-08 06:56 +0800
tags:
  - python
  - artificial intelligence
  - machine learning
  - graph convolutional networks
  - GCN
  - EvolveGCN
  - NBA
  - fan loyalty
description: "Exploring how to use EvolveGCN for NBA fan loyalty analysis, from dynamic graph network theory to practical implementation"
---

NBA fan loyalty is a complex mix of identity, performance, location, and social influence. Research shows that most fans don't switch teams, but some do. Those that switch teams based on performance are called Bandwagon Fans. They are often casual fans supporting teams with recent success. These fans are more likely to switch teams after a team is eliminated or declines. The hardcore fans are usually identity-based and are tied to hometown teams or lifelong affiliations (e.g., passed down through family).

## Understanding Fan Loyalty Through Dynamic Graphs

In our previous post, we explored Graph Convolutional Networks (GCNs) for static graph analysis. However, fan loyalty is inherently dynamic, influenced by:

- Player trades and free agency movements
- Team performance trends
- Championship victories and playoff runs
- Rivalry games and marquee matchups
- Social media engagement patterns

## Introducing EvolveGCN

EvolveGCN extends traditional GCNs to handle dynamic graphs by incorporating recurrent mechanisms. It comes in two main variants:

1. **EvolveGCN-H**: Uses GRU or LSTM to evolve hidden states
2. **EvolveGCN-O**: Directly evolves the GCN's weight matrices

### Mathematical Foundation

Let's break down the EvolveGCN equations in the context of NBA fan networks:

#### Node Representation Update

$$H^{(t+1)} = \text{GCN}(A^{(t)}, X^{(t)}, W^{(t)})$$

- **$H^{(t+1)}$**: The updated node representations at time $t+1$
  - *NBA Context*: This could represent updated fan engagement levels, team affinities, or player popularity metrics after considering the latest games and events.

- **$A^{(t)}$**: The adjacency matrix at time $t$
  - *NBA Context*: Represents connections between entities (fans, teams, players) at a specific time. For example:
    - Fan-to-team connections (who supports which team)
    - Player-to-team affiliations
    - Fan-to-fan interactions on social media

- **$X^{(t)}$**: Node features at time $t$
  - *NBA Context*: A feature vector for each node (fan, team, or player) that could include:
    
    **1. Sentiment Features (for fan nodes):**
    - *Continuous*: A value between -1 (negative) to 1 (positive) representing sentiment polarity
      - Example: `sentiment = 0.75` (very positive), `-0.3` (slightly negative)
    - *Categorical*: One-hot encoded sentiment categories
      - Example: `[1, 0, 0]` for positive, `[0, 1, 0]` for neutral, `[0, 0, 1]` for negative
    - *Multi-dimensional*: Separate scores for different aspects
      - Example: `[team_sentiment, player_sentiment, game_sentiment] = [0.8, -0.2, 0.6]`

    **2. Engagement Features (for fan nodes):**
    - Activity frequency: `[posts_per_week, comments_per_week, likes_per_week]`
    - Recency: `[days_since_last_activity]`
    - Engagement type: `[game_engagement, social_engagement, merchandise_engagement]`

    **3. Team Performance Features (for team nodes):**
    - Game stats: `[win_rate, points_per_game, point_differential]`
    - Recent form: `[wins_last_10, point_diff_last_5]`
    - Playoff status: `[games_behind, magic_number]`

    **4. Player Features (for player nodes):**
    - Performance: `[points, rebounds, assists, player_efficiency]`
    - Social metrics: `[follower_count, engagement_rate]`
    - Contract: `[years_remaining, salary]`

    **5. Temporal Features (for all nodes):**
    - Seasonality: `[day_of_week, month, is_playoffs]`
    - Time-based: `[games_remaining, days_since_last_game]`

    **Example Feature Vector for a Fan Node:**
    ```python
    [
        0.75,        # sentiment_score (-1 to 1)
        5, 2, 12,      # posts, comments, likes (weekly)
        2,             # days_since_last_activity
        0, 1, 0,        # one-hot encoded: [casual, diehard, bandwagon]
        1, 0, 0, 0,     # one-hot encoded: [team_A, team_B, team_C, ...]
        0.8, 0.2, 0.1   # engagement with [games, social, merchandise]
    ]
    ```
    
    **Normalization:**
    - Continuous features are typically normalized (e.g., using Min-Max or Z-score)
    - Categorical features are one-hot encoded or embedded
    - Time-based features might use cyclical encoding for periodic patterns

- **$W^{(t)}$**: The weight matrix at time $t$
  - *NBA Context*: Learns how different features contribute to fan loyalty patterns, adapting as the season progresses.

#### Dynamic Weight Update

$$W^{(t)} = \text{RNN}(W^{(t-1)}, \text{context}^{(t)})$$

- **$\text{RNN}$**: A recurrent neural network (typically GRU or LSTM)
  - *NBA Context*: Captures how the importance of different features evolves over time. For example:
    - Playoff races might increase the importance of recent performance
    - Trade deadlines might increase the weight of roster stability metrics

- **$\text{context}^{(t)}$**: Additional context at time $t$
  - *NBA Context*: Could include:
    - Time since last championship
    - Rivalry week indicators
    - Major events (All-Star break, playoffs)
    - Off-season vs. regular season flags

#### Intuition in NBA Terms

Imagine tracking fan engagement after a major event like the NBA Finals:
1. The adjacency matrix ($A^{(t)}$) captures who's talking to whom about the finals
2. Node features ($X^{(t)}$) include each fan's recent activity and sentiment
3. The GCN propagates this information through the network
4. The RNN updates the model's understanding of what drives loyalty based on this new context
5. The updated weights ($W^{(t)}$) now better predict how fans might react to future events

This dynamic approach allows the model to capture how factors influencing fan loyalty shift throughout the season and in response to specific events.

## Building a Fan Loyalty Model

## Data Processing Pipeline

### 1. Sentiment Analysis from Raw Text

We will be using a pre-trained sentiment analysis model (`pipeline("sentiment-analysis")`)from Hugging Face's model hub.  The model uses the distilbert-base-uncased-finetuned-sst-2-english model.  This means that it is a distilled version of BERT.  It is trained on the Stanford Sentiment Treebank (SST-2).  This model returns labels ("POSITIVE" or "NEGATIVE") with confident scores.
```python
from transformers import pipeline
import numpy as np

def analyze_sentiment(texts):
    # Using a pre-trained sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = sentiment_pipeline(texts)
    
    # Convert to numerical scores (-1 to 1)
    sentiments = []
    for result in results:
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score *= -1
        sentiments.append(score)
    return np.array(sentiments)

# Example usage
fan_comments = [
    "Love how the team is playing this season!",
    "Terrible performance last night, very disappointed.",
    "The new player acquisition is amazing!"
]
sentiments = analyze_sentiment(fan_comments)  # e.g., [0.9, -0.8, 0.95]
```

### 2. Temporal Feature Engineering

```python
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# 1. Time-based features
def extract_temporal_features(df):
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek // 5 == 1
    
    # Cyclical encoding for time features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    return df

# 2. Rolling statistics
def add_rolling_features(df, window=7):
    df['rolling_engagement'] = df['engagement'].rolling(window=window).mean()
    df['engagement_trend'] = df['engagement'].diff(window)
    return df

# 3. Time since last event
def time_since_events(df):
    df['days_since_last_win'] = df['is_win'].cumsum().replace(0, np.nan).ffill()
    return df

# Create a feature engineering pipeline
feature_pipeline = Pipeline([
    ('temporal', FunctionTransformer(extract_temporal_features)),
    ('rolling', FunctionTransformer(add_rolling_features)),
    ('events', FunctionTransformer(time_since_events))
])
```

### 3. Handling Missing Data

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def handle_missing_data(X):
    # Option 1: Simple imputation
    # X.fillna(X.median(), inplace=True)
    
    # Option 2: KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Option 3: Iterative Imputation (more powerful but slower)
    # imputer = IterativeImputer(max_iter=10, random_state=42)
    # X_imputed = imputer.fit_transform(X)
    
    return pd.DataFrame(X_imputed, columns=X.columns)
```

## Model Architecture

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal import EvolveGCNH

class FanLoyaltyModel(torch.nn.Module):
    def __init__(self, node_features):
        super().__init__()
        # EvolveGCNH Parameters:
        # - in_channels: Number of input features per node
        # - out_channels: Size of node embeddings (64 is a common choice, can be tuned)
        # - add_self_loops: Adds self-connections (A + I) to include node's own features
        self.evolve_gcn = EvolveGCNH(
            in_channels=node_features,
            out_channels=64,  # Hidden dimension size
            add_self_loops=True  # Include self-connections in the graph
        )
        # Final classifier layer
        self.classifier = torch.nn.Linear(64, 1)  # Outputs a single probability
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: Node features [num_nodes, num_features]
        # edge_index: Graph connectivity [2, num_edges]
        
        # 1. Get node embeddings
        h = self.evolve_gcn(x, edge_index, edge_weight)  # [num_nodes, 64]
        
        # 2. Predict churn probability (0-1)
        return torch.sigmoid(self.classifier(h))  # [num_nodes, 1]
```

## Model Usage Example

```python
import numpy as np
from torch_geometric.data import Data

# 1. Prepare data
num_nodes = 1000  # Number of fans
num_features = 20  # Number of features per node
num_edges = 5000   # Number of connections

# Generate random features and edges for demonstration
x = torch.randn((num_nodes, num_features))  # Node features
edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Random edges

# 2. Initialize model
model = FanLoyaltyModel(node_features=num_features)

# 3. Make predictions
with torch.no_grad():
    # Get churn probabilities for all nodes
    churn_probs = model(x, edge_index)  # Shape: [num_nodes, 1]
    
    # Convert to numpy for analysis
    churn_probs = churn_probs.numpy().flatten()
    
    # Example: Get top 10 fans most likely to churn
    top_churn_indices = np.argsort(churn_probs)[-10:]
    print(f"Top 10 fans likely to churn: {top_churn_indices}")
    print(f"Their churn probabilities: {churn_probs[top_churn_indices]}")

# 4. Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

def train():
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(x, edge_index)
    
    # Assuming we have ground truth labels (0 or 1)
    # In practice, you'd use your actual labels here
    labels = torch.randint(0, 2, (num_nodes, 1)).float()
    
    # Calculate loss and update
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Understanding the Output:
- The model outputs a single probability value between 0 and 1 for each node
- Values closer to 1 indicate higher probability of churn
- Example output for 5 nodes might look like: `[0.12, 0.87, 0.45, 0.23, 0.91]`
- You can set a threshold (e.g., 0.5) to make binary predictions:
  - `predictions = (churn_probs > 0.5).astype(int)`

### Key Points:
1. **64-dimensional embeddings**: The model learns a 64-dimensional representation of each node that captures complex patterns in the graph structure and node features.
2. **Self-loops**: When `add_self_loops=True`, each node connects to itself, allowing nodes to consider their own features during message passing.
3. **Sigmoid activation**: The final output is passed through a sigmoid to get probabilities between 0 and 1.
4. **Batch processing**: For large graphs, you'd typically process the graph in batches using techniques like neighbor sampling.

### Training Insights

Key findings from our analysis:

1. **Player Impact**: Superstars moving teams cause immediate loyalty shifts
2. **Championship Effects**: Title wins create lasting loyalty boosts
3. **Market Dynamics**: Smaller markets show higher volatility in fan engagement

## Practical Applications

This approach helps teams:
- Predict fan engagement for marketing
- Optimize ticket pricing dynamically
- Personalize fan experiences
- Plan roster moves considering fan impact

## Conclusion

By modeling fan loyalty as a dynamic graph problem, we gain valuable insights into the complex, evolving relationships between teams, players, and fans. The EvolveGCN framework provides a powerful tool for capturing these temporal dynamics.

## References

1. [Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs", 2020](https://arxiv.org/abs/1902.10191)
2. [NBA API Documentation](https://github.com/swar/nba_api)
3. [PyTorch Geometric Temporal Documentation](https://pytorch-geometric-temporal.readthedocs.io/)