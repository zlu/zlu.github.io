---
layout: post
title: '遗传算法：从达尔文到霍兰德'
date: 2025-05-12 18:09 +0800
- Optimization
- Metaheuristics
- Genetic Algorithm
- Python
- Algorithm
comments: true
---

![Genetic Algorithm: Algo-Scope.online](/assets/images/uploads/ga.png)

看动态视觉演示: **[Genetic Algorithm: Algo-Scope.online](https://www.algo-scape.online/ga)**

你是否曾想过，工程师们是如何在复杂的空气动力学约束下优化飞机机翼设计，以达到最佳的燃油效率、升力和阻力平衡的？或者大学是如何安排数千门课程，既要避免课程表冲突，又要确保每门课都有教室的？这些只是遗传算法（Genetic Algorithm, GA）的一些典型应用。遗传算法这个名字的灵感来源于查尔斯·达尔文的**进化论 (evolution)**。

在其开创性的著作《物种起源》中，达尔文阐述了生物在繁殖过程中会发生变异。如果这些变异能让生物体有更好的生存和繁殖机会（即提高其“繁殖适应度 (reproductive fitness)”），那么它们就更有可能被传递给后代。我们通常将此概括为“适者生存 (survival of the fittest)”。

时间快进到1975年。密歇根大学的杰出教授约翰·霍兰德（John Holland）将这些思想加以提炼，并形式化为我们今天所称的遗传算法。在他极具影响力的著作《自然和人工系统中的适应性》(Adaptation in Natural and Artificial Systems) 中，他引入了诸如“选择 (selection)”、“交叉 (crossover)”和“变异 (mutation)”等核心概念，作为“进化”问题解决方案的机制。遗传算法的精妙之处在于，这些自适应系统即使在那些我们不完全理解的复杂环境中，也能够学习并找到优化的解决方案。

遗传算法的核心步骤如下：

1.  **初始化种群 (Initialize Population)：** 我们首先创建一个“种群 (population)”——基本上是一堆针对我们问题的随机潜在解决方案。可以把它想象成把很多想法都扔到墙上，看看哪些能行得通。
    下面是一个如何生成随机位向量（一种常见的解决方案表示方法）的示例：
    ```python
    import random # 确保也为这个片段导入 random 模块

    def generate_random_bit_vector(length):
        vector = []
        for _ in range(length):
            vector.append(1 if random.random() < 0.5 else 0) # 0 或 1 的概率各占一半
        return vector
    ```

2.  **评估适应度 (Assess Fitness)：** 接下来，我们需要一种方法来判断每个解决方案有多“好”。这就是“适应度函数 (fitness function)”发挥作用的地方。它为每个解决方案打分——分数越高，解决方案就越“适应”。

3.  **选择父代 (Selection of Parents)：** 现在到了“适者生存”的环节！我们挑选父代解决方案进行“繁殖”，优先选择那些适应度得分较高的。有几种方法可以做到这一点，比如“适应度比例选择 (Fitness-Proportionate Selection)”（适应度高的个体有更高的被选中概率）或“锦标赛选择 (Tournament Selection)”（随机选几个解决方案比拼，最好的那个成为父代）。

4.  **交叉 (Crossover / Reproduction)：** 这是奇迹发生的地方。我们取两个父代解决方案，混合它们的“基因 (genes)”（解决方案的组成部分），以创建一个或多个“后代 (offspring)”（新的解决方案）。一种常见的方法是“单点交叉 (One-Point Crossover)”：
    ```python
    def one_point_crossover_example(parent1, parent2): # 重命名以避免独立运行时冲突
        # 在解决方案字符串中选择一个随机点
        point = random.randint(1, len(parent1)-1) 
        # 通过交换“尾部”来创建子代
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    ```
    此外还有“两点交叉 (Two-Point Crossover)”（使用两个点交换中间部分）和“均匀交叉 (Uniform Crossover)”（每个位点以一定概率进行交换）。

5.  **变异 (Mutation)：** 为了保持种群的多样性并避免陷入局部最优，我们会向后代中引入随机的改变，即“变异”。这为种群增加了多样性，并有助于探索解空间的新区域。一个简单的“位翻转变异 (Bit-Flip Mutation)”如下：
    ```python
    def bit_flip_mutation_example(vector, prob): # 为清晰起见重命名
        # prob 是变异概率
        return [1 - bit if random.random() < prob else bit for bit in vector]
    ```

6.  **替换与重复 (Replace and Repeat)：** 最后，新一代的后代会取代旧的种群（或其中的一部分）。我们通常会保留迄今为止找到的最佳解决方案（一种称为“精英保留 (elitism)”的策略）。然后，整个循环（步骤2-6）会重复许多代，直到我们找到一个“足够好”的解决方案，或者我们 просто耗尽了时间/迭代次数。

遗传算法的一个经典例子可能不像优化飞机机翼或汽车车身那样激动人心，但它能很好地说明算法的工作原理。我们将优化一个二进制字符串！目标是最大化一个适应度函数：$$f(x) = \text{sum}(x)$$ 其中 $$x$$ 是一个长度为20的二进制字符串。最优解是一个全为1的字符串（其适应度为20）。

让我们通过一些 Python 代码来看看它的实际效果：

```python
import random

# 参数
POP_SIZE = 50       # 种群大小
STRING_LENGTH = 20  # 字符串长度 (个体长度)
MUTATION_PROB = 0.05 # 变异概率
GENERATIONS = 100   # 迭代代数

# 初始化种群 (算法21：生成随机位向量)
def init_population():
    return [[random.choice([0, 1]) for _ in range(STRING_LENGTH)] for _ in range(POP_SIZE)]

# 适应度函数：计算各位之和
def assess_fitness(individual):
    return sum(individual)

# 锦标赛选择 (算法32：锦标赛选择)
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    # 返回锦标赛中适应度最高的个体
    return max(tournament, key=assess_fitness)

# 单点交叉 (算法23：单点交叉)
def one_point_crossover(parent1, parent2):
    # 确保交叉点不在最两端
    point = random.randint(1, STRING_LENGTH - 1) 
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 位翻转变异 (算法22：位翻转变异)
def bit_flip_mutation(individual):
    return [1 - bit if random.random() < MUTATION_PROB else bit for bit in individual]

# 主遗传算法循环 (算法20：遗传算法)
def genetic_algorithm():
    population = init_population()
    overall_best_individual = None
    overall_best_fitness = -1 # 初始化为一个非常低的适应度值

    for generation_num in range(GENERATIONS):
        # 评估当前种群的适应度，并找到当前代的最佳个体
        # 同时，如果出现新的冠军，则更新全局最佳个体
        current_gen_best_fitness = -1
        for individual in population:
            fitness = assess_fitness(individual)
            if overall_best_individual is None or fitness > overall_best_fitness:
                overall_best_individual = list(individual) # 存储一个副本
                overall_best_fitness = fitness
            if fitness > current_gen_best_fitness:
                current_gen_best_fitness = fitness
        
        # 可选：打印当前代的进度
        # print(f"第 {generation_num+1} 代: 最高适应度 = {overall_best_fitness}")

        # 创建新种群
        new_population = []
        # 精英保留：可选择直接将最佳个体（们）带到下一代
        # if overall_best_individual is not None:
        #    new_population.append(list(overall_best_individual)) 

        # 用后代填充种群的其余部分
        # 如果使用精英保留，调整循环范围以保持 POP_SIZE
        for _ in range(POP_SIZE // 2): 
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = one_point_crossover(parent1, parent2)
            new_population.extend([bit_flip_mutation(child1), bit_flip_mutation(child2)])
        
        # 如果 POP_SIZE 是奇数，我们可能会少一个，可以再添加一个变异后的子代或根据需要处理
        if len(new_population) < POP_SIZE and POP_SIZE % 2 != 0:
             parent1 = tournament_selection(population)
             parent2 = tournament_selection(population)
             child1, _ = one_point_crossover(parent1, parent2)
             new_population.append(bit_flip_mutation(child1))

        population = new_population[:POP_SIZE] # 确保种群大小得以维持

        # 如果找到最优解，则提前停止
        if overall_best_fitness == STRING_LENGTH:
            print(f"在第 {generation_num+1} 代找到最优解!")
            break
            
    return overall_best_individual, overall_best_fitness

# 运行遗传算法
best_solution, best_fitness = genetic_algorithm()
print(f"找到的最佳解决方案: {best_solution}, 适应度: {best_fitness}")
```
这就是遗传算法的简要介绍！从达尔文的观察到霍兰德的公式化，遗传算法为解决复杂的优化问题提供了一种强大且出人意料的直观方法。所以下次当你看到一个效率极高的设计或一个完美优化的时间表时，很有可能遗传算法（或者非常类似它的东西）在其中发挥了作用（或者我们应该说，是它的“基因”在发挥作用）！