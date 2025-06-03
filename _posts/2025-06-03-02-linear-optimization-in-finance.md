---
layout: post
title: "Linear Optimization in Finance: Portfolio Allocation and Solvers - Part 2"
date: 2025-06-02
comments: true
tags:
  - python
  - artificial intelligence
  - machine learning
  - finance
  - linear optimization
  - CBC
  - PuLP
description: "Linear Optimization in Finance: Portfolio Allocation and Solvers - Part 2"
---

In the last post, we looked at the liearity in finance. We implicitly assumed that linearity can solve CAPM and APT pricing models.  But what if the numbe rof securities increases and certain limitations appear?  How might one handle such a constraint-based system?  Linear optimization shines here because it focuses on minimizing (such as volatility) or maximizing (such as profit) an objective function under given constraints.  In this post, we explore how linear optimization techniques power portfolio allocation and other financial decisions. We'll see how Python and open-source solvers like CBC make these methods accessible.

## Linear Optimization

Linear optimization helps in portfolio allocation by maximizing or minimizing an objective function under given constraints.

### Example Problem
Maximize: $ f(x, y) = 3x + 2y $
Subject to:
- $ 2x + y \leq 100 $
- $ x + y \leq 80 $
- $ x \leq 40 $
- $ x, y \geq 0 $

Again, we can use PuLP (`pip install pulp`) to solve this problem.  PuLP is a linear programming library that provides optimization solvers such as (CBC, GLPK, CPLEX, Gurobi, etc.).  One strength is that it allows you to define optimization problems in mathematical notation.

For example, to define a decision variables:
```python
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)
```
Where `x` and `y` are called decision variables, which we want to optimize.  `lowBound=0` ensures that both varialbes are non-negative.  We usually use such variables for portfolio weights.

To define a optimization problem, we do:
```python
problem = pulp.LpProblem("Maximization Problem", pulp.LpMaximize)
```
`pulp.LpMaximize` means we want to maximize the objective function.  We can also use `pulp.LpMinimize` to minimize the objective function.

Next we setup an object function as such:
```python
problem += 3*x + 2*y, "Objective Function"
```
That means we want to maximize f(x, y) = 3x + 2y.  For example, this can be used to maximize return or profit margins.

Then we want to add some constraints in the system.
```python
problem += 2*x + y <= 100, "Constraint 1"
problem += x + y <= 80, "Constraint 2"
problem += x <= 40, "Constraint 3"
```
This means that the sum of the weights of the two assets must be less than or equal to 100, and the sum of the weights of the two assets must be less than or equal to 80.  We can also add more constraints to the system.

Finally we call `.solve()` to solve the problem.
```python
problem.solve()
```

Putting everything together we have:

```python
problem += 3*x + 2*y, "Objective Function"
problem += 2*x + y <= 100, "Constraint 1"
problem += x + y <= 80, "Constraint 2"
problem += x <= 40, "Constraint 3"
```

To define a problem:

```python
# Python Implementation
import pulp

# Define variables
x = pulp.LpVariable("x", lowBound=0)
y = pulp.LpVariable("y", lowBound=0)

# Define problem
problem = pulp.LpProblem("Maximization Problem", pulp.LpMaximize)
problem += 3*x + 2*y, "Objective Function"
problem += 2*x + y <= 100, "Constraint 1"
problem += x + y <= 80, "Constraint 2"
problem += x <= 40, "Constraint 3"

# Solve problem
problem.solve()

# Print results
for variable in problem.variables():
    print(f"{variable.name} = {variable.varValue}")
```

    Welcome to the CBC MILP Solver 
    Version: 2.10.3 
    Build Date: Dec 15 2019 
    ...
    x = 20.0
    y = 60.0


### Understanding Linear Optimization in Detail

Linear optimization (also called Linear Programming or LP) is a mathematical method for finding the best outcome in a mathematical model whose requirements are represented by linear relationships. It's widely used in finance for portfolio optimization, resource allocation, and risk management.

#### Mathematical Foundation

A linear programming problem has the standard form:

**Objective Function:**
$$\text{Maximize (or Minimize): } c^T x = c_1x_1 + c_2x_2 + \ldots + c_nx_n$$

**Subject to constraints:**
$$Ax \leq b \text{ (inequality constraints)}$$
$$Ax = b \text{ (equality constraints)}$$
$$x \geq 0 \text{ (non-negativity constraints)}$$

Where:
- $x = [x_1, x_2, \ldots, x_n]^T$ are the decision variables
- $c = [c_1, c_2, \ldots, c_n]^T$ are the objective function coefficients
- $A$ is the constraint matrix
- $b$ is the right-hand side vector

#### Key Properties of Linear Programming

1. **Linearity**: Both objective function and constraints are linear
2. **Feasible Region**: The set of all points satisfying all constraints
3. **Optimal Solution**: Located at vertices (extreme points) of the feasible region
4. **Convexity**: The feasible region is always convex
5. **Duality**: Every LP problem has an associated dual problem


### The CBC MILP Solver: Comprehensive Overview

**CBC (Coin-or Branch and Cut)** is an open-source Mixed Integer Linear Programming (MILP) solver developed by the COIN-OR (Computational Infrastructure for Operations Research) project. It's one of the most widely used solvers for optimization problems.

#### What is CBC MILP Solver?

**CBC** stands for:
- **C**oin-or
- **B**ranch and
- **C**ut

**MILP** stands for:
- **M**ixed
- **I**nteger
- **L**inear
- **P**rogramming

CBC can solve:
1. **Linear Programming (LP)**: All variables are continuous
2. **Integer Programming (IP)**: All variables are integers
3. **Mixed Integer Programming (MIP)**: Some variables are integers, others continuous
4. **Binary Programming**: Variables are restricted to 0 or 1

#### How CBC Works: The Branch-and-Cut Algorithm

CBC uses a sophisticated **Branch-and-Cut** algorithm:

1. **Linear Relaxation**: First solves the problem ignoring integer constraints
2. **Branching**: If solution has fractional values for integer variables, creates sub-problems
3. **Cutting Planes**: Adds additional constraints to tighten the relaxation
4. **Bounding**: Uses bounds to eliminate sub-problems that can't improve the solution
5. **Pruning**: Eliminates branches that are infeasible or suboptimal

```python
# Comprehensive Linear Programming Example with Visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pulp

# Define the problem from our previous example
# Maximize: 3x + 2y
# Subject to: 2x + y <= 100, x + y <= 80, x <= 40, x,y >= 0

def plot_linear_programming_problem():
    """
    Visualize the linear programming problem and its solution
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create a grid of points
    x = np.linspace(0, 50, 400)
    y = np.linspace(0, 100, 400)
    X, Y = np.meshgrid(x, y)
    
    # Define constraints
    # 2x + y <= 100 => y <= 100 - 2x
    # x + y <= 80 => y <= 80 - x
    # x <= 40
    # x, y >= 0
    
    # Plot constraint lines
    x_line = np.linspace(0, 50, 100)
    
    # Constraint 1: 2x + y <= 100
    y1 = 100 - 2*x_line
    ax1.plot(x_line, y1, 'r-', linewidth=2, label='2x + y ≤ 100')
    
    # Constraint 2: x + y <= 80
    y2 = 80 - x_line
    ax1.plot(x_line, y2, 'b-', linewidth=2, label='x + y ≤ 80')
    
    # Constraint 3: x <= 40
    ax1.axvline(x=40, color='g', linewidth=2, label='x ≤ 40')
    
    # Non-negativity constraints
    ax1.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='k', linewidth=1, alpha=0.5)
    
    # Find feasible region vertices
    vertices = []
    
    # Intersection points
    # Origin
    vertices.append((0, 0))
    
    # x-axis intersection with constraint 1: 2x + 0 = 100 => x = 50, but limited by x <= 40
    vertices.append((40, 0))
    
    # y-axis intersection with constraint 1: 0 + y = 100 => y = 100, but limited by x + y <= 80
    vertices.append((0, 80))
    
    # Intersection of constraints 1 and 2: 2x + y = 100 and x + y = 80
    # Solving: x = 20, y = 60
    vertices.append((20, 60))
    
    # Intersection of constraint 2 and x = 40: 40 + y = 80 => y = 40
    vertices.append((40, 40))
    
    # Filter vertices that satisfy all constraints
    feasible_vertices = []
    for x_val, y_val in vertices:
        if (x_val >= 0 and y_val >= 0 and 
            2*x_val + y_val <= 100 and 
            x_val + y_val <= 80 and 
            x_val <= 40):
            feasible_vertices.append((x_val, y_val))
    
    # Plot feasible region
    if feasible_vertices:
        # Sort vertices to form a proper polygon
        feasible_vertices = sorted(feasible_vertices)
        # Add the vertex (0, 80) and (20, 60) in correct order
        ordered_vertices = [(0, 0), (40, 0), (40, 40), (20, 60), (0, 80)]
        
        polygon = Polygon(ordered_vertices, alpha=0.3, facecolor='yellow', 
                         edgecolor='black', linewidth=2)
        ax1.add_patch(polygon)
        
        # Mark vertices
        for i, (x_val, y_val) in enumerate(ordered_vertices):
            ax1.plot(x_val, y_val, 'ko', markersize=8)
            ax1.annotate(f'({x_val}, {y_val})', (x_val, y_val), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Plot objective function contours
    objective_values = [120, 150, 180, 210]
    for obj_val in objective_values:
        # 3x + 2y = obj_val => y = (obj_val - 3x)/2
        y_obj = (obj_val - 3*x_line) / 2
        ax1.plot(x_line, y_obj, '--', alpha=0.7, 
                label=f'3x + 2y = {obj_val}' if obj_val == 180 else '')
    
    # Mark optimal solution
    ax1.plot(20, 60, 'r*', markersize=15, label='Optimal Solution (20, 60)')
    
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Linear Programming Problem Visualization', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Evaluate objective function at each vertex
    vertex_values = []
    for x_val, y_val in ordered_vertices:
        obj_value = 3*x_val + 2*y_val
        vertex_values.append((x_val, y_val, obj_value))
    
    # Create bar chart of objective values at vertices
    vertices_labels = [f'({x}, {y})' for x, y, _ in vertex_values]
    obj_values = [obj for _, _, obj in vertex_values]
    
    bars = ax2.bar(range(len(vertices_labels)), obj_values, 
                   color=['red' if obj == max(obj_values) else 'skyblue' for obj in obj_values],
                   alpha=0.8)
    
    ax2.set_xlabel('Vertices', fontsize=12)
    ax2.set_ylabel('Objective Function Value (3x + 2y)', fontsize=12)
    ax2.set_title('Objective Function Values at Vertices', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(vertices_labels)))
    ax2.set_xticklabels(vertices_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, obj_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return vertex_values

# Generate the visualization
vertex_analysis = plot_linear_programming_problem()

print("Vertex Analysis:")
print("=" * 40)
for x, y, obj_val in vertex_analysis:
    print(f"Vertex ({x:2.0f}, {y:2.0f}): Objective Value = {obj_val:3.0f}")

optimal_vertex = max(vertex_analysis, key=lambda x: x[2])
print(f"\nOptimal Solution: ({optimal_vertex[0]}, {optimal_vertex[1]}) with value {optimal_vertex[2]}")
```

![Linear Programming Problem Visualization](/assets/images/uploads/lp/lp.png)


#### CBC Solver Advantages and Applications in Finance

**Advantages:**
1. **Open Source**: Free to use, no licensing costs
2. **Robust**: Well-tested and reliable
3. **Versatile**: Handles LP, IP, MIP, and BIP problems
4. **Efficient**: Uses state-of-the-art algorithms
5. **Integration**: Works well with Python (PuLP), R, and other languages

**Financial Applications:**
1. **Portfolio Optimization**: Asset allocation with constraints
2. **Risk Management**: VaR optimization, stress testing
3. **Trading Strategies**: Order execution optimization
4. **Capital Budgeting**: Project selection with budget constraints
5. **Asset-Liability Management**: Matching assets and liabilities
6. **Derivatives Pricing**: Optimal hedging strategies
7. **Credit Risk**: Loan portfolio optimization
8. **Operational Research**: Branch location, staffing optimization

## Integer Programming
Integer programming involves optimization where some or all variables are constrained to be integers.

### Example Problem
Minimize costs for purchasing contracts from dealers with constraints on quantities and costs.

```python
# Python Implementation
import pulp

# Define dealers and costs
dealers = ["X", "Y", "Z"]
variable_costs = {"X": 500, "Y": 350, "Z": 450}
fixed_costs = {"X": 4000, "Y": 2000, "Z": 6000}

# Define variables
quantities = pulp.LpVariable.dicts("quantity", dealers, lowBound=0, cat=pulp.LpInteger)
is_orders = pulp.LpVariable.dicts("orders", dealers, cat=pulp.LpBinary)

# Define problem
model = pulp.LpProblem("Cost Minimization Problem", pulp.LpMinimize)
model += sum([variable_costs[i]*quantities[i] + fixed_costs[i]*is_orders[i] for i in dealers]), "Minimize Costs"
model += sum([quantities[i] for i in dealers]) == 150, "Total Contracts Required"
model += 30 <= quantities["X"] <= 100, "Boundary of X"
model += 30 <= quantities["Y"] <= 90, "Boundary of Y"
model += 30 <= quantities["Z"] <= 70, "Boundary of Z"

# Solve problem
model.solve()

# Print results
for variable in model.variables():
    print(f"{variable.name} = {variable.varValue}")
```

    Welcome to the CBC MILP Solver 
    Version: 2.10.3 
    ...
    orders_X = 0.0
    orders_Y = 0.0
    orders_Z = 0.0
    quantity_X = 0.0
    quantity_Y = 90.0
    quantity_Z = 60.0

---

This is a simple example of an integer programming problem.  In this case, we want to minimize the cost of purchasing contracts from dealers.  We have constraints on the total number of contracts and the quantity of contracts from each dealer.  We also have constraints on the quantity of contracts from each dealer.

In the next post, we will look at how CBC can be used to solve more complex problems in finance.


*Next: A Deeper Look into CBC in Finance →* 