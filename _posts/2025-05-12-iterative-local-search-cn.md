---
layout: post
title: "探索优化：迭代局部搜索（ILS）指南"
date: 2025-05-12 15:30 +0800
tags: [优化, 元启发式算法, 迭代局部搜索, ILS, Python, 算法]
comments: true
---

![迭代局部搜索]({{ site.url }}/assets/images/uploads/ils.png)

查看 ILS 的可视化演示：[AlgoScope](https://www.algo-scape.online/optimization)

许多现实世界的问题，从物流到调度，都涉及到从大量可能性中寻找最佳解决方案。这些被称为优化问题。虽然找到绝对的*最佳*解决方案（全局最优解）对于复杂场景来说可能计算成本高昂甚至不可能，但元启发式算法提供了强大的策略，可以在合理的时间内找到非常好的解决方案。

在这篇博文中，我们将探讨一种流行的元启发式算法：迭代局部搜索（ILS）。我们将分解其核心概念，查看伪代码，并给出一个简单的 Python 实现。

## 迭代局部搜索（ILS）简明解释

想象一下你蒙着眼睛爬山。你从一个随机的地点开始，试图找到最高点。你向上迈步，直到无法再爬得更高。此时，你到达了一个“局部”山顶（一个局部最优解）。然而，你无从知晓附近或远处是否还有其他更高的山峰。这个初始过程就是“局部搜索”。

为了找到可能更高的山峰（一个更好的解决方案），迭代局部搜索引入了一个“扰动”步骤。你实质上是进行一次“跳跃”，到达一个不同的、有些随机的坐标，希望落入一个新的区域，在那里进行另一次局部搜索可能会引导你到达一个更高的山顶。你多次重复这个局部搜索和扰动的过程，以探索解空间的重要性。

### ILS 伪代码

ILS 算法的通用结构如下所示：

```
function ILS():
solution = GenerateInitialSolution()
    best_solution = LocalSearch(solution) // 对初始解应用局部搜索    

    while not StoppingConditionMet(): // 例如：时间限制、迭代次数
        // 扰动当前最优解以跳出局部最优
        perturbed_solution = Perturb(best_solution) 
        // 对扰动后的解应用局部搜索
        locally_optimal_solution = LocalSearch(perturbed_solution)

        // 如果新解更好，则接受它 (假设是最小化问题)
        if Cost(locally_optimal_solution) < Cost(best_solution):
            best_solution = locally_optimal_solution

    return best_solution
```
其中：
-   `GenerateInitialSolution()`: 创建一个起始点。这可以是随机的，也可以基于简单的启发式方法（例如，贪婪算法）。它不保证最优性，但有助于算法探索解空间的各个部分。
-   `LocalSearch(solution)`: 通过进行小的、局部的更改来改进给定的解决方案。例如，在旅行商问题（TSP）中，这可能涉及交换两个城市的顺序；在车间调度问题中，它会重新分配作业。
-   `Perturb(solution)`: 比 `LocalSearch` 更显著地修改解决方案，以逃离搜索空间的当前区域并探索新的区域。
-   `Cost(solution)`: 评估解决方案的质量（例如，TSP 中的总路径长度，作业调度中的完工时间）。目标通常是最小化此成本。
-   `StoppingConditionMet()`: 确定何时终止算法（例如，固定的迭代次数、时间限制，或者当解决方案质量在一段时间内没有改善时）。

现在让我们看一个 ILS 在 Python 中的可能实现，我们的目标函数是最大化：

$$ f(x) = - (x - 3)^2 + 9 $$

视觉上它看起来像这样：
```
f(x)
 9 |                       *
 8 |                  *       *
 7 |              *             *
 6 |          *                   *
 5 |      *                         *
 4 |  *                               *
    +------------------------------------→ x
      0      1      2      3      4      5
```      

我们选择这个简单的抛物线函数是因为它在 x=3 处有一个唯一的峰值（最大值为9）。因此，很容易可视化搜索算法在其上移动时的工作方式。

所以基于该算法，我们有以下步骤：

```
- 从一个随机点开始。
- 执行局部搜索（例如，贪婪爬山法），直到没有更多改进。这是我们初始的 best_solution。
- 扰动 best_solution 以获得一个新的起点。
- 对这个新的扰动解执行局部搜索。
- 如果这个新的局部最优解优于我们总体的 best_solution，则更新 best_solution。
- 重复步骤3-5，直到达到停止条件（例如，时间预算）为止。
```	

一个 Python 实现是：

```python
import random

# 需要最大化的目标函数
def objective_function(x):
    return -(x - 3)**2 + 9

# 局部搜索：简单的爬山法
def local_search(current_x, step_size=0.1):
    # 对于最大化问题，我们寻找更大的值
    while True:
        fx_current = objective_function(current_x)
        fx_plus = objective_function(current_x + step_size)
        fx_minus = objective_function(current_x - step_size)

        if fx_plus > fx_current:
            current_x += step_size
        elif fx_minus > fx_current:
            current_x -= step_size
        else:
            break # 没有改进，找到了局部最优解
    return current_x

# 扰动：进行一次随机跳跃
def perturb(current_x, perturbation_strength=1.0):
    return current_x + random.uniform(-perturbation_strength, perturbation_strength)

# 迭代局部搜索
def ils(initial_x, max_iterations=100, step_size_local_search=0.1, perturbation_strength_ils=1.0):
    # 初始局部搜索
    best_x = local_search(initial_x, step_size=step_size_local_search)
    best_fx = objective_function(best_x)

    print(f"初始 (第一次局部搜索后): x={best_x:.3f}, f(x)={best_fx:.3f}")

    for i in range(max_iterations):
        perturbed_x = perturb(best_x, perturbation_strength=perturbation_strength_ils) # 从目前找到的最佳解开始扰动
        locally_optimal_x = local_search(perturbed_x, step_size=step_size_local_search)
        current_fx = objective_function(locally_optimal_x)

        if current_fx > best_fx: # 对于最大化问题
            best_x = locally_optimal_x
            best_fx = current_fx
            print(f"迭代 {i+1} (新的最优解): x={best_x:.3f}, f(x)={best_fx:.3f}")
        # else:
            # print(f"迭代 {i+1}: x={locally_optimal_x:.3f}, f(x)={current_fx:.3f} (未改进)")
            
    return best_x, best_fx

# 运行 ILS
x0 = random.uniform(0, 6) # 随机起始点
x_found_ils, fx_ils = ils(x0)
print(f"\nILS 在 x = {x_found_ils:.3f} 处找到最大值, f(x) = {fx_ils:.3f}")

# 示例输出:
# 初始 (第一次局部搜索后): x=0.800, f(x)=4.160
# 迭代 1 (新的最优解): x=3.000, f(x)=9.000
# ... (对于这个简单函数，后续迭代可能会探索但找不到更好的解)
# ILS 在 x = 3.000 处找到最大值, f(x) = 9.000
```

## 示例1：作业调度
作业调度，有时也称为车间调度问题（Job Shop Problem），是 ILS 的一个经典应用。
假设你有5台机器和20个作业。每个作业都必须分配给一台机器，并且根据机器的不同，每个作业花费的时间也不同。目标是将作业分配给机器，以使完成所有作业的总时间（完工时间）最小化。

ILS 算法可以是：

1.	从随机分配作业到机器开始（初始解决方案）。
2.	应用局部搜索：贪婪地重新分配作业以减少总时间（例如，将作业从过载的机器移动到欠载的机器，或在机器之间交换作业），直到无法进行局部改进。这是初始的 best_solution。
3.	扰动：当陷入局部最小值时，更大幅度地随机交换或重新分配一些作业，以创建一个新的起点。
4.  对这个扰动后的解决方案应用局部搜索。
5.  如果结果优于当前的 best_solution，则更新 best_solution。
6.	重复步骤3–5，直到达到定义的迭代次数或时间。


## 示例2：旅行商问题（TSP）
经典的 NP-hard 问题 TSP 指的是一个销售员需要访问一定数量的城市，每个城市恰好访问一次，然后必须返回起始城市。目标是找到可能的最短行程。

ILS 算法可以是：


1.	从随机的城市顺序（路径）开始作为初始解决方案。
2.	应用局部搜索：使用像 2-opt（如果能减少总路径长度，则交换路径中两条不相邻的边）或 3-opt 这样的操作符，重复进行直到找不到进一步的改进。这是初始的 best_solution。
3.	扰动：一旦陷入困境，打乱路径的一部分或执行几次“双桥”移动（比 2-opt 更显著的变化），以创建一条新的起始路径。
4.  对这条新的扰动路径应用局部搜索。
5.  如果得到的路径比当前的 best_solution 短，则更新 best_solution。
6.	重复步骤3-5。

