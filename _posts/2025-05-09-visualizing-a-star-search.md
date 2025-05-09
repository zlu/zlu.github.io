---
layout: post
title: Visualizing A-Star Search
date: 2025-05-09 09:21 +0800
comments: true
categories: [Data Structure, Javascript, Visualization]
---

![A* Search](/assets/images/uploads/a-star.png)

[Live A* Demo](https://algo-scape.online/astar)

## A* Search Simplied Explained

A* is a popular heuristic-based search algorithm that finds the shortest path from a start node to a target node. It combines the strengths of Dijkstra's algorithm and Greedy Best-First Search by considering both the cost to reach a node and the estimated cost to the goal.

Imagine you’re using a map to get from your house to a coffee shop. There are many possible roads you could take, but you want the shortest or fastest one.

A* search works like this:
1.	It starts at your current location.
2.	It looks at all the places you could go next.
3.	For each possible route, it calculates:
  - How far you’ve already gone (called the “cost so far”).
  - How far it thinks it is to the goal (this is an estimate, called a “heuristic”).
4.	It adds those two numbers to decide which route looks the most promising.
5.	It keeps repeating this process—always choosing the path with the best combined score—until it reaches the goal.

So, A* is like a smart traveler that not only looks at where they’ve been but also tries to guess which future path looks shortest, and keeps adjusting as they go.

## The Algorithm

The algorithm uses the following formula to evaluate nodes:

$$ f(n) = g(n) + h(n) $$

Where:
- `g(n)`: The cost to reach the current node `n` from the start node.
- `h(n)`: The heuristic estimate of the cost to reach the goal from node `n`.
- `f(n)`: The total estimated cost of the path through node `n`.

Essentially, the algorithm chooses the path of the lowest cost by combinng the cost traversed so far (g(n)) and the cost from the current node to reach the final destination (h(n)).  h(n) is calculated using Manhattan Distance between the two nodes.

$$ h(n) = |x1 - x2| + |y1 - y2| $$

Where:
- \( (x1, y1) \) are the coordinates of the first node.
- \( (x2, y2) \) are the coordinates of the second node.
- \( h(n) \) is the heuristic value representing the Manhattan Distance.

This promises the follow characteristics of A*:

1. **Optimality**: A* guarantees the shortest path if the heuristic function `h(n)` is admissible (never overestimates the cost).
2. **Completeness**: A* will always find a solution if one exists.
3. **Heuristic Function**: The choice of `h(n)` significantly impacts the algorithm's performance.

### A-Star Implementation:

1. Start at the initial node A1.
2. Look at its neighbors.
3. For each neighbor:
  - Calculate g(n) = g(current) + 1
  - Calculate h(n) = manhattan distance to goal
  - Calculate f(n) = g(n) + h(n)
4. Add neighbors to the "open set" (a priority list of where to explore next).
5. From the open set, pick the node with the smallest f(n)
6. Repeat until the goal is reached.


### Step-by-Step Execution of A*

Suppose we have the follow setup where S is the starting point, G is the goal, '#' are the barriers.  We want to find the shortest path from S to G using A*.

```
S . . . .
# # . # .
. . . # .
. # # # .
. . . . G
```

Where:

- S = Start
- G = Goal
- '#' = Wall (not passable)
- . = Open path

#### **Step 1: Start at \( S = (0, 0) \)**
- \( g(S) = 0 \)
- \( h(S) = |0 - 4| + |0 - 4| = 8 \)
- \( f(S) = g(S) + h(S) = 0 + 8 = 8 \)

**Frontier**: \( [(0, 0)] \)  
**Visited**: \( \emptyset \)

---

#### **Step 2: Expand \( (0, 0) \)**
- Neighbors of \( (0, 0) \): \( (0, 1) \) (open), \( (1, 0) \) (wall, skip).
- For \( (0, 1) \):
  - \( g(0, 1) = g(0, 0) + 1 = 1 \)
  - \( h(0, 1) = |0 - 4| + |1 - 4| = 7 \)
  - \( f(0, 1) = g(0, 1) + h(0, 1) = 1 + 7 = 8 \)

**Frontier**: \( [(0, 1)] \)  
**Visited**: \( [(0, 0)] \)

---

#### **Step 3: Expand \( (0, 1) \)**
- Neighbors of \( (0, 1) \): \( (0, 2) \) (open), \( (1, 1) \) (wall, skip), \( (0, 0) \) (already visited, skip).
- For \( (0, 2) \):
  - \( g(0, 2) = g(0, 1) + 1 = 2 \)
  - \( h(0, 2) = |0 - 4| + |2 - 4| = 6 \)
  - \( f(0, 2) = g(0, 2) + h(0, 2) = 2 + 6 = 8 \)

**Frontier**: \( [(0, 2)] \)  
**Visited**: \( [(0, 0), (0, 1)] \)

---

#### **Step 4: Expand \( (0, 2) \)**
- Neighbors of \( (0, 2) \): \( (0, 3) \) (open), \( (1, 2) \) (open), \( (0, 1) \) (already visited, skip).
- For \( (0, 3) \):
  - \( g(0, 3) = g(0, 2) + 1 = 3 \)
  - \( h(0, 3) = |0 - 4| + |3 - 4| = 5 \)
  - \( f(0, 3) = g(0, 3) + h(0, 3) = 3 + 5 = 8 \)
- For \( (1, 2) \):
  - \( g(1, 2) = g(0, 2) + 1 = 3 \)
  - \( h(1, 2) = |1 - 4| + |2 - 4| = 5 \)
  - \( f(1, 2) = g(1, 2) + h(1, 2) = 3 + 5 = 8 \)

**Frontier**: \( [(0, 3), (1, 2)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2)] \)

---

#### **Step 5: Expand \( (0, 3) \)**
- Neighbors of \( (0, 3) \): \( (0, 4) \) (open), \( (1, 3) \) (wall, skip), \( (0, 2) \) (already visited, skip).
- For \( (0, 4) \):
  - \( g(0, 4) = g(0, 3) + 1 = 4 \)
  - \( h(0, 4) = |0 - 4| + |4 - 4| = 4 \)
  - \( f(0, 4) = g(0, 4) + h(0, 4) = 4 + 4 = 8 \)

**Frontier**: \( [(1, 2), (0, 4)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3)] \)

---

#### **Step 6: Expand \( (1, 2) \)**
- Neighbors of \( (1, 2) \): \( (2, 2) \) (open), \( (1, 1) \) (wall, skip), \( (0, 2) \) (already visited, skip).
- For \( (2, 2) \):
  - \( g(2, 2) = g(1, 2) + 1 = 4 \)
  - \( h(2, 2) = |2 - 4| + |2 - 4| = 4 \)
  - \( f(2, 2) = g(2, 2) + h(2, 2) = 4 + 4 = 8 \)

**Frontier**: \( [(0, 4), (2, 2)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)] \)

---

#### **Step 7: Expand \( (0, 4) \)**
- Neighbors of \( (0, 4) \): \( (1, 4) \) (open), \( (0, 3) \) (already visited, skip).
- For \( (1, 4) \):
  - \( g(1, 4) = g(0, 4) + 1 = 5 \)
  - \( h(1, 4) = |1 - 4| + |4 - 4| = 3 \)
  - \( f(1, 4) = g(1, 4) + h(1, 4) = 5 + 3 = 8 \)

**Frontier**: \( [(2, 2), (1, 4)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2)] \)

---

#### **Step 8: Expand \( (2, 2) \)**
- Neighbors of \( (2, 2) \): \( (3, 2) \) (wall, skip), \( (2, 3) \) (open).
- For \( (2, 3) \):
  - \( g(2, 3) = g(2, 2) + 1 = 5 \)
  - \( h(2, 3) = |2 - 4| + |3 - 4| = 3 \)
  - \( f(2, 3) = g(2, 3) + h(2, 3) = 5 + 3 = 8 \)

**Frontier**: \( [(1, 4), (2, 3)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 2)] \)

---

#### **Step 9: Expand \( (1, 4) \)**
- Neighbors of \( (1, 4) \): \( (2, 4) \) (open), \( (0, 4) \) (already visited, skip).
- For \( (2, 4) \):
  - \( g(2, 4) = g(1, 4) + 1 = 6 \)
  - \( h(2, 4) = |2 - 4| + |4 - 4| = 2 \)
  - \( f(2, 4) = g(2, 4) + h(2, 4) = 6 + 2 = 8 \)

**Frontier**: \( [(2, 3), (2, 4)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 2), (1, 4)] \)

---

#### **Step 10: Expand \( (2, 4) \)**
- Neighbors of \( (2, 4) \): \( (3, 4) \) (open), \( (2, 3) \) (already visited, skip).
- For \( (3, 4) \):
  - \( g(3, 4) = g(2, 4) + 1 = 7 \)
  - \( h(3, 4) = |3 - 4| + |4 - 4| = 1 \)
  - \( f(3, 4) = g(3, 4) + h(3, 4) = 7 + 1 = 8 \)

**Frontier**: \( [(2, 3), (3, 4)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 2), (1, 4), (2, 4)] \)

---

#### **Step 11: Expand \( (3, 4) \)**
- Neighbors of \( (3, 4) \): \( (4, 4) \) (goal), \( (2, 4) \) (already visited, skip).
- For \( (4, 4) \):
  - \( g(4, 4) = g(3, 4) + 1 = 8 \)
  - \( h(4, 4) = |4 - 4| + |4 - 4| = 0 \)
  - \( f(4, 4) = g(4, 4) + h(4, 4) = 8 + 0 = 8 \)

**Frontier**: \( [(4, 4)] \)  
**Visited**: \( [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 2), (1, 4), (2, 4), (3, 4)] \)

---

#### **Step 12: Expand \( (4, 4) \)**
- Goal reached!

---

### Final Path:
The shortest path from \( S \) to \( G \) is:
```plaintext
(0, 0) → (0, 1) → (0, 2) → (0, 3) → (0, 4) → (1, 4) → (2, 4) → (3, 4) → (4, 4)
```


### Example Implementation in Javascript
You may find it helpful to play with a live demo here: [Live A* Demo](https://algo-scape.online/astar)

Feel free to change the start, goal, grid size and even define your own barriers!

```typescript
export function AStar(
    start: [number, number],
    end: [number, number],
    obstacles: Set<string>,
    terrain: number[][]
): { path: number[][]; visited: number[][] } {
    const openSet = new Set<string>();
    const cameFrom: { [key: string]: [number, number] } = {};
    const gScore: { [key: string]: number } = {};
    const fScore: { [key: string]: number } = {};
    const allVisitedNodes: number[][] = [];

    const getKey = (x: number, y: number) => `${x},${y}`;

    // Initialize the start node
    const startKey = getKey(start[0], start[1]);
    openSet.add(startKey);
    gScore[startKey] = 0;
    fScore[startKey] = heuristic(start, end);
    allVisitedNodes.push(start.slice());

    while (openSet.size > 0) {
        const current = getLowestFScore(openSet, fScore);
        const currentKey = getKey(current[0], current[1]);

        if (current[0] === end[0] && current[1] === end[1]) {
            return { path: reconstructPath(cameFrom, current), visited: allVisitedNodes };
        }

        openSet.delete(currentKey);

        const neighbors = getNeighbors(current, terrain);
        for (const neighbor of neighbors) {
            const neighborKey = getKey(neighbor[0], neighbor[1]);
            if (obstacles.has(neighborKey)) continue;

            const tentativeGScore = (gScore[currentKey] || 0) + getDistance(current, neighbor, terrain);

            if (!gScore[neighborKey] || tentativeGScore < (gScore[neighborKey] || Infinity)) {
                cameFrom[neighborKey] = current;
                gScore[neighborKey] = tentativeGScore;
                fScore[neighborKey] = tentativeGScore + heuristic(neighbor, end);

                if (!openSet.has(neighborKey)) {
                    openSet.add(neighborKey);
                    allVisitedNodes.push(neighbor.slice());
                }
            }
        }
    }

    return { path: [], visited: allVisitedNodes };
}
```

