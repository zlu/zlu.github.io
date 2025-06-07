---
layout: post
title: "分类与逻辑回归"
date: 2025-06-04
comments: true
tags:
  - python
  - 人工智能
  - 机器学习
  - 逻辑回归
  - 分类
  - 数学
description: "分类与逻辑回归"
---

线性回归和逻辑回归其实比你想象的更相似 :)
它们都是所谓的参数模型。让我们先看看什么是参数模型，以及它们与非参数模型的区别。

## 线性回归 vs 逻辑回归

- 线性回归：用于回归问题的线性参数模型。
- 逻辑回归：用于分类问题的线性参数模型。

## 参数回归模型：

- 假设函数形式
  - 模型假设特定的数学方程（如线性、多项式）。
  - 例：在线性回归中，模型假设如下形式：
    $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon$<br>
    其中 $\beta_i$ 是参数，$\epsilon$ 是误差项。
- 参数数量固定
- 模型复杂度由一组固定参数决定，与训练数据量无关。
- 学习高效
- 由于模型结构预先定义，训练参数模型通常计算高效。
- 可解释性强
- 许多参数模型（如线性回归）具有可解释性，便于理解每个特征对预测的影响。
- 用有限参数总结数据的模型。
- 对数据分布有假设。
  - 如线性/逻辑回归、神经网络

## 非参数模型：

- 无法用有限参数描述的模型。
- 对数据分布无假设。
  - 如基于实例的学习，使用训练数据生成假设
- 例子：kNN 和决策树

## 逻辑（Sigmoid）函数

$$f(x) = \frac{1}{1 + e^{-x}}$$

其导数为：
$$f'(x) = f(x)(1 - f(x))$$

这说明：

- 任意点的导数取决于该点的函数值
- 当 f(x) 接近 0 或 1 时，导数变得很小
- 当 f(x) = 0.5（Sigmoid 曲线中点）时导数最大

这个性质使得 sigmoid 在机器学习中很有用：

- 它将输入"压缩"到 [0,1] 区间
- 导数易于计算（只需输出乘以 1 减自身）
- 在 0 和 1 附近导数很小，有助于防止权重更新过大

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)
```

## 多元 sigmoid 函数

$$ f(x, y) = \frac{1}{1 + e^{-(ax + by + c)}} $$

其中 $a$、$b$、$c$ 是决定 sigmoid 曲面形状的常数。

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义二维 sigmoid 函数
def sigmoid(x, y, a=1, b=1, c=0):
    return 1 / (1 + np.exp(-(a*x + b*y + c)))

# 生成网格点
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = sigmoid(X, Y)

# 绘制曲面
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Sigmoid(X, Y)')
ax.set_title('多元 Sigmoid 函数的 3D 图')

plt.show()

"""
Meshgrid 用于生成 x 和 y 的取值。
- Z 值通过 sigmoid 函数计算。
- 使用 plot_surface() 和 viridis 颜色映射绘制 3D 曲面。
"""
```

![png](/assets/images/uploads/logistic-regression_files/logistic-regression_4_0.png)

## 交叉熵方法

交叉熵衡量两个概率分布之间的差异。在机器学习中，它常作为损失函数，尤其用于分类问题。

关键概念：

### 基本公式

$H(p,q) = -\sum_x p(x)\log(q(x))$

其中：

- $p(x)$ 为真实概率分布（标签）
- $q(x)$ 为预测概率分布
- $\sum_x$ 表示对所有可能的 x 求和

### 二元交叉熵（用于二分类）：

$H(y,\hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

其中：

- $y$ 为真实标签（0 或 1）
- $\hat{y}$ 为预测概率

```python
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

### 性质

- 总是非负
- 预测与真实分布完全一致时为 0
- 值越大表示预测越差
- 对自信但错误的预测惩罚更重

交叉熵在分类中的优势：

- 比均方误差提供更强梯度
- 适用于概率输出（如 sigmoid 或 softmax）
- 对过度自信的错误预测惩罚大

## 分类模型

### 简单模型：

### 感知机（Perceptron）

感知机是一种二分类器，通过对输入特征加权求和并通过激活函数判断类别。

#### 感知机包括：

1. 输入（特征）：$x_1, x_2, …, x_n$
2. 权重：$w_1, w_2, …, w_n$（训练中学习）
3. 偏置（b）：调整决策边界
4. 求和函数：$z = \sum w_i x_i + b$
5. 激活函数：通常为阶跃函数，输出：
   - $1$ 若 $z \geq 0$（一类）
   - $0$ 若 $z < 0$（另一类）

#### 数学表达

$y = f\left(\sum_{i=1}^{n} w_i x_i + b \right)$

其中 f(z) 为激活函数（常用阶跃或符号函数）。

#### 感知机学习算法

1. 随机初始化权重和偏置。
2. 对每个训练样本：
   - 用加权和和激活函数计算输出。
   - 与真实标签比较，按如下公式更新权重：
     $w_i = w_i + \eta (y_{\text{true}} - y_{\text{pred}}) x_i$
   - 类似地调整偏置。
3. 重复直到分类正确或达到停止条件。

#### 局限性

- 感知机只能分类线性可分数据（如 AND、OR 逻辑门）。
- 无法解决如 XOR 这类需多层的非线性问题。

#### 多层感知机（MLP）

为克服线性可分限制，将多个感知机堆叠成多层感知机（MLP），并用非线性激活函数（如 sigmoid、ReLU、tanh）。

#### 示例：用感知机实现 AND 门分类

我们将训练感知机模拟 AND 逻辑门。

#### 数据集（AND 门）

| 输入 $x_1$ | 输入 $x_2$ | 输出 $y$ |
| :--------: | :--------: | :------: |
|     0      |     0      |    0     |
|     0      |     1      |    0     |
|     1      |     0      |    0     |
|     1      |     1      |    1     |

感知机应学会正确预测输出。

#### 感知机模型

感知机公式：

$y = f(w_1 x_1 + w_2 x_2 + b)$

其中：

- $w_1$, $w_2$ 为权重
- $b$ 为偏置
- $f(z)$ 为阶跃激活函数：
  $$
  f(z) =
  \begin{cases}
  1, & \text{if } z \geq 0 \\
  0, & \text{if } z < 0
  \end{cases}
  $$

#### 学习算法

1. 初始化权重 $w_1$, $w_2$ 和偏置 $b$（如小随机值）。
2. 对每个训练样本 $(x_1, x_2, y)$：
   - 计算 $z = w_1 x_1 + w_2 x_2 + b$
   - 应用激活函数得预测 $\hat{y}$。
   - 若预测错误，按如下公式更新权重：
     $w_i = w_i + \eta (y_{\text{true}} - \hat{y}) x_i$<br>  
     其中 $\eta$ 为学习率。
3. 重复直到权重收敛。

#### 决策边界可视化

让我们可视化训练好的 AND 感知机：

1. 感知机找到一条分隔数据的直线。
2. 由于 AND 可线性分割，感知机可解决。
3. 决策边界是一条直线。

![perceptron](/assets/images/uploads/logistic-regression_files/perceptron.png)

上图展示了感知机对 AND 门的分类：

- 红圈（类别 0）为 AND 输出为 0 的点。
- 蓝方块（类别 1）为 AND 输出为 1 的点。
- 虚线为感知机学到的决策边界。

虚线下方为 0，上方为 1。由于 AND 可线性分割，单层感知机可正确分类。

### K 近邻（KNN）

KNN 是一种非参数、基于实例的学习方法，可用于**分类**和**回归**。它根据最近邻的多数类别（分类）或均值（回归）做预测。

#### 步骤：

- 选择 K：确定考虑的最近邻数量。
- 计算距离：用如下度量计算新点与所有训练点的距离：
  - 欧氏距离：
    $ d(A,B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $
  - 曼哈顿距离
  - 闵可夫斯基距离
- 找到 K 个最近邻：确定距离新点最近的 K 个点。
- 做出预测：
  - 分类：取 K 邻居中最多的类别（多数投票）。
  - 回归：取 K 邻居的均值。

想象你有一组数据，要将新点分为红或蓝：

- 若 K = 3，取最近 3 个邻居。
- 若 2 个为蓝，1 个为红，则新点归为蓝。

KNN 优点：

- 简单易实现
- 无需训练（惰性学习）
- 适合小数据集
- 可用于分类和回归

KNN 缺点：

- 计算量大（大数据集时每次预测都要算距离）
- 对无关特征和噪声敏感
- K 的选择很关键

KNN 适用场景：

- 小到中等规模数据集
- 需要可解释性
- 作为复杂模型前的基线

如何选 K：

- K 小 → 邻域小 → 高复杂度 → 易过拟合
- K 大 → 邻域大 → 低复杂度 → 易欠拟合
- 实践中常选 3–15，或 K < 𝑁（N 为训练样本数）。

```python
# KNN 示例实现
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap

# 生成合成数据集
# n_features=2: 2 个特征。
# n_informative=2: 两个特征都有用。
# n_redundant=0: 无冗余特征。
# n_clusters_per_class=1: 每类一个簇。
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42)
# 训练 KNN 模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# 创建决策边界的网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# 对网格每个点做预测
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(6, 4))
cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#0000FF"])
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

# 绘制数据点
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=50)
plt.legend(handles=scatter.legend_elements()[0], labels=["类别 0", "类别 1"])

# 标签
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title("K 近邻决策边界 (K=5)")
plt.show()

# 下图为 KNN 决策边界说明：
# 红/蓝区域：两类的决策边界。
# 不同颜色的点：不同类别的数据点。
# 平滑边界：KNN 根据 K 个邻居的多数类别分配标签。
```

![png](/assets/images/uploads/logistic-regression_files/logistic-regression_9_0.png)

### 混淆矩阵

混淆矩阵用于评估分类模型的性能。它是一个表格，汇总了模型的预测与实际结果的对比。

- 真正例（TP）：被正确预测为正类的样本数。
- 真负例（TN）：被正确预测为负类的样本数。
- 假正例（FP）：被错误预测为正类的样本数（I 型错误）。
- 假负例（FN）：被错误预测为负类的样本数（II 型错误）。

混淆矩阵有助于理解模型的表现，尤其是区分不同类别的能力。它揭示了模型的错误类型，并可用于计算准确率、精确率、召回率和 F1 分数等指标。

1. **坐标轴**：

- **X 轴（列）**：模型预测的类别。
- **Y 轴（行）**：数据集的真实类别。

2. **结构**：

- 每个单元格显示实际类别与预测类别组合的样本数。
- 对角线（左上到右下）为正确预测。

4. **解读**：

- TP 和 TN 多说明模型表现好。
- FP 和 FN 多说明模型有改进空间。

在代码中，混淆矩阵常用热力图可视化，便于观察分布。`annot=True` 参数确保每个单元格显示具体数值。

![roc-auc](/assets/images/uploads/logistic-regression_files/roc-auc.png)

ROC-AUC 分数是用于评估二分类模型质量的指标。
ROC 代表受试者工作特征曲线（Receiver Operating Characteristic curve）。
AUC 代表曲线下的面积（Area Under the Curve）。

- ROC 曲线绘制不同阈值下的真正率（TPR）与假正率（FPR）。
- AUC（曲线下的面积）用单一数值总结模型在所有阈值下的表现。

真正率（TPR）：TP / (TP + FN) → 也叫召回率
假正率（FPR）：FP / (FP + TN)

- AUC 范围 0 到 1：
- 1.0 = 完美分类器
- 0.5 = 随机猜测
- <0.5 = 比随机还差（系统性误分）

应用场景：

- 二分类问题
- 尤其适用于：
- 类别不平衡（如欺诈检测、疾病诊断）
- 关注预测排序而非具体标签

重要性：

- 衡量模型区分不同类别的能力。
- 在类别不平衡时比准确率更稳健。
- 可跨阈值比较模型。

本教程到此结束，其他常用分类模型如 SVM、决策树、随机森林等将在后续介绍。

如有问题或反馈，欢迎联系我。

感谢阅读！
