---
layout: post
title: "缺失的部分 - 轻松入门 fastai 和 PyTorch - 第一部分"
date: 2025-05-24
comments: true
categories: 
  - 机器学习
  - 神经网络
tags:
  - 感知机
  - 神经网络
  - 深度学习
  - 机器学习
  - 二元分类
  - 算法
  - python
  - 人工智能
  - 机器学习基础
  - fastai
  - PyTorch
description: "了解感知机——神经网络的基本构建模块。包括实用的 Python 实现、可视化解释，以及垃圾邮件检测等真实应用。非常适合机器学习初学者。"
permalink: /machine-learning/perceptrons-neural-networks-building-block/
---

## 介绍
Fastai 是一个流行的机器学习库，构建在 PyTorch 之上。
对有些人来说，它是纯粹的福音，因为它承担了繁琐的设置和大量样板代码的工作。
但对另一些人来说，它则是"伪装很深的福音"，因为：
1. 看似简单的语句背后有很多逻辑，理解起来并不直观。
2. 变量命名如果不了解背后的技术概念，很难理解。
3. 命名的简写会让不熟悉的概念变得更加晦涩。

你当然可以死记硬背如何定义一个模型，甚至是训练和预测的简单流程。但乐趣也就到此为止了。
一旦遇到有独特性的实际问题，照搬现有例子就不再适用。当出现错误时（编程中 100% 会发生），就很难修正。

在这个系列中，我会尝试为读者拆解这些内容，希望大多数概念能变得更加清晰。

## 本地环境搭建
我推荐在本地环境中动手实践。当然，遇到大任务时我们也可以用免费的 Google Colab，但有一个可以随时查看和操作的本地环境总是好的。

这里分享一下我的个人配置——一台依然很好用但有些年头的 Macbook Pro M1。

Apple Silicon (M1/M2) 对 CUDA（NVIDIA 专用）支持有限，但 PyTorch 为 M1 提供了原生的 MPS（Metal Performance Shaders）后端以实现 GPU 加速。
- 从 PyTorch 1.12+ 开始，MPS 后端支持在 M1/M2 GPU 上训练。
- 一旦配置好 PyTorch，fastai 也能正常运行。

M1 GPU 跑 PyTorch 的局限：
- MPS 仍处于实验阶段，并非所有功能都支持。
- 有些操作可能会回退到 CPU。
- 内存使用比独立显卡更受限。

所以，如果你有以下需求建议用 Colab：
- 训练大型模型，如 ResNet50、ViT、LLM。
- 需要仅支持 CUDA 的操作。
- 想免费用高端 GPU。

### M1 本地环境安装步骤
建议用 Conda 环境（miniforge 或 miniforge3 比 Anaconda 更适合 ARM 架构 Mac）：

- 步骤 1：安装 miniforge（如果还没装）
	- 访问 https://github.com/conda-forge/miniforge 并安装适用于 ARM64 的 miniforge3
- 步骤 2：创建环境
	- conda create -n fastai-m1 python=3.10
	- conda activate fastai-m1
- 步骤 3：安装支持 MPS 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

- 注意：如果你用的是 macOS 12.3+，也可以尝试 MPS 后端：
 - pip install torch torchvision torchaudio
- 步骤 4：测试 MPS 是否可用：
	- python -c "import torch; print(torch.backends.mps.is_available())"
- 步骤 5：安装 fastai
	- pip install fastai


```python
# 另一种方式是用 python 代码验证环境
import torch
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
```

    MPS available: True
    MPS built: True



```python
# 训练一个小型玩具模型
from fastai.vision.all import *

# 加载著名的 mnist（手写数字识别）数据集
# 每张图片是一个手写数字，比如 0、1、2、3。
# 数据集以目录结构存储，包含验证集（mnist_sample/valid）
# 和训练集（mnist_sample/train）
# 子目录 3/ 里是数字 3 的图片；7/ 里是数字 7 的图片
# 这是图像分类任务常见的数据格式，每个类别（这里是每个数字）
# 都有自己的目录，里面放着属于该类别的图片。模型会在训练过程中
# 学会区分这两个数字（3 和 7）。
path = untar_data(URLs.MNIST_SAMPLE)
print("Main path:", path)
print("\nSubdirectories:")
for root, dirs, files in os.walk(path):
    level = root.replace(str(path), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level == 0:  # 只显示根目录下的文件
        for f in files:
            print(f"{indent}    {f}")
```

    Main path: /Users/zlu/.fastai/data/mnist_sample
    
    Subdirectories:
    mnist_sample/
        labels.csv
      valid/
        7/
        3/
      train/
        7/
        3/



```python
import pandas as pd
# 查看标签数据
labels_df = pd.read_csv(path/'labels.csv')
print("\nlabels (first 5 rows):")
print(labels_df.head())
```

    
    labels (first 5 rows):
                    name  label
    0   train/3/7463.png      0
    1  train/3/21102.png      0
    2  train/3/31559.png      0
    3  train/3/46882.png      0
    4  train/3/26209.png      0



```python
# 从训练集中随机取一张图片
train_path = path/'train'
img_files = list((train_path/'7').ls())

# Python Imaging Library (PIL) 是一个用于打开、处理和保存多种图片格式的库。
# 它的一个分支叫 Pillow 被广泛使用。PIL 已经包含在 fastai.vision.all 里。
img = PILImage.create(img_files[0])
print(f"这是一张手写数字 7")
img.show(); 
# 如果没有分号，会额外打印出 "<Axes: >"。这是 matplotlib 的内容，PIL 用它来显示图片。
# 它是为绘图自动创建的默认坐标轴对象。
```

    这是一张手写数字 7


    
![png](/assets/images/uploads/fastai/tutorial_4_1.png)
    



```python
dls = ImageDataLoaders.from_folder(path, valid='valid')
```

dls 代表 DataLoaders。
	• 它是对两个 PyTorch DataLoader 的封装：
	• 一个用于训练（dls.train）
	• 一个用于验证（dls.valid）

fastai 提供 .from_folder 方法，可以根据如下目录结构快速创建：
```
    path/
    ├── train/
    │   ├── 3/
    │   ├── 7/
    ├── valid/
        ├── 3/
        ├── 7/
````


```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(1) # 1 表示只训练 1 个 epoch（即完整遍历一遍训练集）
```



<style>
    /* 关闭部分样式 */
    progress {
        /* 去除 Firefox 和 Opera 的默认边框 */
        border: none;
        /* Safari polyfill 需要 */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.298345</td>
      <td>0.176359</td>
      <td>0.934740</td>
      <td>00:13</td>
    </tr>
  </tbody>
</table>




<style>
    /* 关闭部分样式 */
    progress {
        /* 去除 Firefox 和 Opera 的默认边框 */
        border: none;
        /* Safari polyfill 需要 */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.055849</td>
      <td>0.031771</td>
      <td>0.989205</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>


ResNet
ResNet-18 来自《深度残差学习用于图像识别》<https://arxiv.org/abs/1512.03385>。
ResNet18 是一个预训练模型，最初在 ImageNet（包含 1000 个类别的通用图片）上训练。

fine_tune() 方法是 fastai 的便捷方法，实现了特定的训练策略：
1. 首先冻结预训练层，只训练新的分类头
2. 然后解冻所有层，训练整个模型
3. 自动调整学习率和其他训练参数
这就是"微调"之名的由来——我们用一个预训练模型（ResNet18）
并将其适配到我们的具体任务（识别手写数字 3 和 7）。

"head"
在深度学习和迁移学习中，"head"指的是神经网络最后用于特定任务预测的层。

在本例中：
ResNet18 是一个在 ImageNet（大规模通用图片数据集）上预训练的模型
"head"是我们在 ResNet18 顶部新加的分类层，使其适用于我们的任务（分类数字 3 和 7）

可以这样理解：

ResNet18 的主体（"骨干"）已经学会了从图片中提取有用特征
"head"就像我们加在顶部的新"大脑"，用来解释这些特征以完成具体任务

"只训练新的分类头"意味着：
保持所有预训练的 ResNet18 层冻结（不变）
只训练我们新加的最后几层，用于数字分类

这是迁移学习中常见的策略，因为：
预训练骨干已经能提取有用特征
我们只需训练 head，让它能解释这些特征以完成具体任务
这比从头训练整个网络快得多，也需要更少的数据


```python
learn.show_results(max_n=6, figsize=(4, 4))
```



<style>
    /* 关闭部分样式 */
    progress {
        /* 去除 Firefox 和 Opera 的默认边框 */
        border: none;
        /* Safari polyfill 需要 */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](/assets/images/uploads/fastai/tutorial_9_2.png)
    


在每张数字图片上方，有两个绿色数字。第一个是真实值（手写数字），第二个是预测值。

## 总结

本教程我们介绍了：
- 如何搭建 fastai 的本地环境
- 如何加载和预处理 MNIST 数据集
- 如何构建一个简单的图像分类模型
- 如何训练和评估模型
- 如何用训练好的模型进行预测

下期我们会探索更复杂的话题。敬请期待，周六愉快！
