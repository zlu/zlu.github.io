---
layout: post
title: "无监督学习"
date: 2025-06-01
comments: true
tags:
  - python
  - 人工智能
  - 机器学习
  - 无监督学习
  - 分类
  - 聚类
  - k-means
  - pca
  - 特征缩放
  - 数学
description: "无监督学习"
---

无监督学习是机器学习的一个分支，其数据没有明确的标签或目标。目标是发现数据中的隐藏模式、分组或结构。与有监督学习（模型从带标签的样本中学习）不同，无监督学习处理的是原始、无标签的数据。

## 为什么要用无监督学习？
- **无标签数据丰富：** 现实世界中大多数数据都是无标签的（如图片、文本、传感器数据）。
- **探索性数据分析：** 有助于在应用其他方法前理解数据的结构和分布。
- **数据压缩：** 降低维度以便存储、可视化或加速计算。
- **降噪：** 去除无关或冗余特征。
- **预处理：** 通过提取有用特征或降噪提升有监督学习的效果。
- **新颖/异常检测：** 发现异常模式或离群点。
- **生成建模：** 学习生成与输入数据相似的新样本。

## 无监督学习的类型
- **聚类：** 将相似的数据点分组（如客户细分）。
- **降维：** 在保留重要信息的前提下减少特征数量（如 PCA、t-SNE）。
- **异常检测：** 识别罕见或异常的数据点（如欺诈检测）。
- **关联规则挖掘：** 发现变量间有趣的关系（如购物篮分析）。
- **密度估计：** 估计数据的概率分布（如高斯混合模型）。
- **生成建模：** 学习生成新数据（如 GAN、VAE）。

---

## 有监督 vs. 无监督学习
- **有监督学习：**
    - 带标签观测：每个观测是 (x, y) 的元组，x 为特征向量，y 为输出标签，两者通过未知函数 f(x) = y 关联。
    - 训练时：利用带标签观测学习 x 与 y 的关系，即找到最适合观测的函数（或模型）h(x)
    - 目标：确保学到的模型 h(x) 能准确预测未见过的测试输入的输出标签（泛化能力）
    - 标签：训练时是"老师"，测试时是"验证者"
- **无监督学习：**
    - 无标签特征向量数据集
    - 目标：在无明确标签的情况下发现数据中的结构、模式或分组

---

## 聚类

聚类是将一组对象分组的任务，使得同一组（簇）中的对象彼此更相似，而不同组之间的对象差异更大。

- **目标：** 找到观测/对象/特征向量之间的自然分组
- **应用：**
    - 市场细分
    - 社交网络分析
    - 文档或图片组织
    - 异常检测
    - 推荐系统

### 常见聚类算法

#### K-Means（K均值）
- 通过最小化组内方差将数据分为 k 个簇。
- 快速且可扩展，但需指定 k，且假设簇为球形。
- 对初始化和异常值敏感。

#### 层次聚类
- 通过合并（自底向上）或分裂（自顶向下）簇构建聚类树（树状图）。
- 不需要提前指定簇数。
- 能捕捉嵌套簇，但大数据集下扩展性较差。

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# 生成合成数据
np.random.seed(42)
X = np.random.rand(10, 2)

# 执行层次聚类
Z = linkage(X, 'ward')

# 绘制树状图
plt.figure(figsize=(6, 3))
dendrogram(Z)
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_3_0.png)

#### DBSCAN（基于密度的空间聚类）
- 将密集点分为一组，将孤立点标记为离群点。
- 不需要指定簇数。
- 能发现任意形状的簇，对异常值鲁棒。

```python
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据
X = np.random.rand(100, 2)

# 应用 DBSCAN
db = DBSCAN(eps=0.1, min_samples=5).fit(X)
labels = db.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title('DBSCAN 聚类')
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_5_0.png)

#### 高斯混合模型（GMM）
- 假设数据由多个高斯分布混合生成。
- 能拟合椭圆形簇，提供软分配（概率）。
- 适用于密度估计和聚类。

```python
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# 生成合成数据
X = np.random.rand(300, 2)

# 拟合 GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('高斯混合模型聚类')
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_15_0.png)

---

## 相似性度量（距离指标）

用于量化任意两个特征向量之间关系的强度。
- 连续值特征
    - 例如 x = (0.1, 11, 15, 1.5)，可计算任意两点的"距离"

### 欧氏距离
测量 n 维空间中两点的最短直线距离。可看作多维毕达哥拉斯定理。

$D_{\text{euclidean}}(a, b) = \sqrt{ \sum_{i=1}^{n} (a_i - b_i)^2 }$

```python
import numpy as np

def euclidean_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))
```

### 曼哈顿距离

也叫 L1 范数。就像出租车在网格城市中只能水平或垂直行驶。

$D_{\text{manhattan}}(a, b) = \sum_{i=1}^{n} |a_i - b_i|$

```python
def manhattan_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.sum(np.abs(a - b))
```

### 切比雪夫距离
度量任一坐标上的最大差值。适用于某一方向主导代价的场景：

$D_{\text{chebyshev}}(a, b) = \max_{i=1}^{n} |a_i - b_i|$

```python
def chebyshev_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.max(np.abs(a - b))
```

### 汉明距离

汉明距离是指两个等长字符串（或数组）在对应位置上不同的个数。

若 $a$ 和 $b$ 是长度为 $n$ 的向量：

$D_{\text{hamming}}(a, b) = \sum_{i=1}^{n} \mathbb{1}(a_i \neq b_i)$

其中 $\mathbb{1}$ 是指示函数，不同为 1，相同为 0。

适用于：
- 二进制向量（如 [0, 1, 1, 0]）
- 字符串（如 "karolin" vs "kathrin"）

### 汉明距离矩阵

汉明距离矩阵是一个对称矩阵，显示多个向量之间的两两汉明距离。

示例：

假设有如下二进制向量：
```
A = [1, 0, 1, 1]
B = [1, 1, 0, 1]
C = [0, 0, 1, 0]
```
则汉明距离矩阵为：
```
	A	B	C
A	0	2	2
B	2	0	3
C	2	3	0
```
每个值表示对应行列向量不同的位数。

下例中，scipy.spatial.distance.hamming 返回归一化距离（比例），需乘以向量长度得实际距离。

```python
from scipy.spatial.distance import hamming
import numpy as np

# 二进制数据
X = np.array([
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 1, 0]
])

# 计算两两汉明距离
n = len(X)
hamming_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        hamming_matrix[i, j] = hamming(X[i], X[j]) * X.shape[1]  # 乘以长度得实际距离

print(hamming_matrix)
```

    [[0. 2. 2.]
     [2. 0. 4.]
     [2. 4. 0.]]

---

## K-Means（示例）

- 随机初始化 k 个质心。
- 将每个点分配到最近的质心。
- 根据分配重新计算质心。
- 重复直到收敛。

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成二维 3 簇合成数据
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 应用 KMeans 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 可视化聚类
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.title("K-Means 聚类")
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_17_0.png)

---

## 降维

降维技术将高维数据转换为低维空间，同时尽量保留信息。常用于可视化、降噪和加速后续算法。

### 主成分分析（PCA）
- 线性方法，将数据投影到最大方差方向（主成分）。
- 适用于可视化和降噪。

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 生成合成数据
np.random.seed(42)
X = np.random.rand(100, 5)

# 降到二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA: 二维投影')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_19_0.png)

### t-SNE（t-分布随机邻域嵌入）
- 非线性方法，将高维数据可视化到 2 或 3 维。
- 保留局部结构，适合可视化聚类。

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 生成合成数据
np.random.seed(42)
X = np.random.rand(100, 10)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title('t-SNE 可视化')
plt.show()
```

![png](/assets/images/uploads/unsupervised-learning_files/unsupervised-learning_21_1.png)

### UMAP（统一流形近似与投影）
- 非线性降维与可视化方法。
- 在许多情况下比 t-SNE 更好地保留局部和全局结构。

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# 生成合成数据
np.random.seed(42)
X = np.random.rand(100, 10)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title('UMAP 可视化')
plt.show()
```

---

## 评估指标

### 聚类评估
- **轮廓系数（Silhouette Score）：** 衡量样本与本簇和其他簇的相似度。范围 -1 到 1，越高越好。
- **Davies-Bouldin 指数：** 越低聚类效果越好。
- **调整兰德指数、互信息：** 用于与真实标签对比（如有）。

```python
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 生成合成数据
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# 拟合 KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# X 和 labels 都有 300 个样本
score = silhouette_score(X, labels)
print(f'轮廓系数: {score:.2f}')
```

### 降维评估
- **解释方差（PCA）：** 选定主成分捕获的方差比例。
- **可视化：** 绘制降维后的数据，观察是否保留了簇或结构。

---

## 特征缩放

特征缩放是将输入变量（特征）变换到相似尺度的过程。

例如：
- 原始特征值：[年龄: 25, 薪资: 90,000, 身高: 180]
- 缩放后可能变为：[0.2, 0.8, 0.7]

### 为什么要特征缩放？
许多机器学习算法假设特征在同一尺度——否则可能：
- 模型偏向（大数值特征主导）
- 基于梯度的算法收敛慢
- 距离计算不准确（如 k-NN、k-means、SVM 等）

对尺度敏感的算法：
- k-NN
- k-Means
- SVM
- PCA
- Logistic/Linear Regression（带正则化）
- 神经网络

### 最小-最大归一化（Rescaling）

将值缩放到固定区间，通常为 [0, 1]。

公式：

$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$

示例：
若身高范围 150–200 cm，某人 180 cm：

$\frac{180 - 150}{200 - 150} = \frac{30}{50} = 0.6$

优点：
- 保持原分布形状
- 保证所有值在区间内

缺点：
- 对异常值敏感（极端值会压缩其他值）

### Z-score 标准化（Standard Scaling）

将特征中心化为 0，方差为 1——适用于近似高斯分布的数据。

公式：

$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$

其中：
- $\mu$ = 特征均值
- $\sigma$ = 特征标准差

示例：
若均值为 40，标准差为 10，30 岁：

$\frac{30 - 40}{10} = -1$

优点：
- 无界，但比最小-最大法更能处理异常值
- 适合假设高斯分布的算法（如线性模型、逻辑回归、PCA）

---

## 真实世界中的无监督学习应用
- **客户细分：** 按购买行为分组以便精准营销。
- **文档聚类：** 按主题组织新闻或论文。
- **图像压缩：** 通过提取关键特征减少图片大小。
- **异常检测：** 识别欺诈交易或网络入侵。
- **推荐系统：** 找到相似用户或物品以个性化推荐。
- **基因组学：** 从基因表达数据中发现疾病亚型。

---

## 可视化建议
- 用散点图展示 2D/3D 降维结果（PCA、t-SNE、UMAP），便于观察聚类。
- 按聚类标签或已知类别着色。
- 层次聚类用树状图。
- 高维数据可用成对图或平行坐标图。

---

## 参考与延伸阅读
- [scikit-learn: Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)
- [Pattern Recognition and Machine Learning by Bishop]
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron]
- [Distill.pub: Visual essays on ML](https://distill.pub/) 