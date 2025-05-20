---
layout: post
title: Diffusion Optimization - Optimizing Smart Energy Grid in a City
date: 2025-05-19
comments: true
description: "A practical example of diffusion optimization in smart energy grid optimization in a city"
keywords: "diffusion optimization, smart energy grid, privacy-preserving optimization, decentralized learning, machine learning"
categories: [Machine Learning, Optimization, Privacy]
---

Diffusion and Stochastic Gradient Descent (SGD) optimization are fundamentally
different optimization paradigms. SGD is considered to be the gold standard,
leveraging classical optimization theory, while diffusion provides a
decentralized process providing privacy protection. This characteristic makes
diffusion optimization well-suited for a wide array of real-world applications,
especially considering in this AI-world, privacy is becoming an increasing
concern for society.

Let's start with an example of a city smart energy grid and examine how
well-suited diffusion is. Suppose the city has 100 smart homes, each is
equipped with solar panels, smart meters, energy storage units, and a local
processor. Each home (we call them an agent) wants to optimize its energy usage
model. This means that they need to know how to store, consume or sell the
energy back to the grid. But here comes the privacy issue. Private information
about each home cannot be shared across the network, meaning there isn't a
central controller. The homes form a peer-to-peer network, where they
communicate only with neighbors.

Each home (agent) $i$ would look at its current model $\theta_i^t$, compute how
much the model could be improved using its own data. Then it takes a step in
the opposite direction of the gradient to minimize the loss, which in turn
produces an intermediate result $\psi_i^{t+1}$, before combining it with
neighbors. Mathematically it looks like this:

$$ \psi_i^{t+1} = \theta_i^t - \eta \nabla L_i(\theta_i^t) $$

Where:

- $L_i$ is the loss function representing the cost of energy usage vs.
  price and availability. $\nabla$ means the gradient of the local loss function.
  Putting them together, $\nabla L_i(\theta_i^t)$ is the gradient of the loss
  function at $\theta_i^t$. It represents how the local model at node $i$ should
  change to reduce its own error.
- $\psi_i^{t+1}$ is the intermediate model for agent $i$ after its local update at
  time $t+1$. This is not yet combined with neighbors' models. Thus it is a
  preliminary update based only on local information.
  $\theta_i^t$ is the current model parameter vector of agent $i$ at time $t$. It
  represents its current knowledge or state.
- $\eta$ is learning rate, which is usually a positive scalar that controls the
  step size of the update. A higher $\eta$ leads to larger changes.

Next is the diffusion step, where the optimization gets its name from, akin to
how heat diffuses (spreads) itself. In this step, each agent $j$ computes its
own local update $\psi_j^{t+1}$, home $i$ would then collect these updates,
apply weights $a_{i,j}$ to it, and then averages them to compute its new model
$\theta_i^{t+1}$. This means a diffusion of information since each agent moves toward agreement with its neighbors. Mathematically:

$$\theta_i^{t+1} = \sum_{j \in \mathcal{N}_i} a_{ij} \psi_j^{t+1}$$

Where:

- $\mathcal{N}_i$ is the set of neighboring agents of $i$.
- $\sum_{j \in \mathcal{N}_i} \cdots$ is the sum over all neighbors $j$ of agent
  $i$.
- $a_{i,j}$ is the combination weight that agent $i$ assigns to agent $j$'s
  intermediate model. These weights usually satisfy:
  $$
  \sum_{j \in \mathcal{N}_i} a_{i,j} = 1,\quad a_{ij} \geq 0
  $$
Meaning that the weights assigned to neighbors by agent $i$ sums to 1,
ensuring a proper convex combination. Also each individual weight $a_{i,j}$
must be positive to avoid negative influence.

![Diffusion Optimization - Home Topology of Smart Grid](/assets/images/uploads/zlu-me-diffusion-optimization-smart-grid.png)

In conclusion, diffusion models offer a decentralized paradigm, allowing all
homes to converge to a global optimum, purely via local computation and
communication. No raw data such as energy usage leaves each home and only model
updates are shared. This architecture also allows new homes to join the network
without requiring a central configuration. In case of a single home failure, no
other homes will be affected. It is thus a privacy-aware, scalable, and
resilient architecture.


