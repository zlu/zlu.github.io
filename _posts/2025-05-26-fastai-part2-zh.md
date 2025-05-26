---
layout: post
title: "图像分类最新指南 - Fastai 第二部分"
date: 2025-05-26
comments: true
categories: 
  - 机器学习
tags:
  - python
  - 人工智能
  - 机器学习
  - fastai
  - PyTorch
description: "图像分类最新指南 - Fastai 第二部分"
---
在本指南中，我们将介绍如何获取图像数据并用其训练模型。

## 获取图片
曾经，Bing 搜索 API 被用来获取图片。但该 API 将于 2025 年 8 月下线。目前，有一些替代方式可以获取图片：

1. DuckDuckGo 图片搜索（非官方）：无需注册或 API key。由于是非官方接口，图片质量无法保证，且如果 DuckDuckGo 更改搜索引擎，接口可能失效。

```python
from duckduckgo_search import ddg_images

results = ddg_images("golden retriever", max_results=10)
for r in results:
    print(r["image"])
```

2. [SerpAPI](https://serpapi.com/)，这是一个 Google/Bing/Yahoo 的封装。提供免费额度，超出后需付费。

3. [Unsplash API](https://unsplash.com/developers)：提供高质量、免版权图片。需要 API key。小规模使用免费。

4. [Pexels](https://www.pexels.com/api/)：提供大量免费图库照片。需要 API key。小规模使用免费。

5. [Pixabay](https://pixabay.com/api/docs/)：提供大量免费图库照片和视频。需要 API key。小规模使用免费。

6. [Flickr](https://www.flickr.com/services/api/)：提供大量免费图库照片和视频。需要 API key。小规模使用免费。

你可以根据图片需求和规模选择合适的方式。但本指南将使用 [Kaggle 数据集](https://www.kaggle.com/datasets)。

## 下载数据集

首先，我们需要用 Kaggle CLI 下载 Stanford Dogs 数据集到 Notebooks 目录。

```bash
# 创建/激活合适的虚拟环境并安装 kaggle-cli
pip install kaggle
# 然后需要从你的 Kaggle 账户获取 kaggle.json 文件
mkdir -p ~/.kaggle
mv /path/to/your/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d jessicali9530/stanford-dogs-dataset
```
这会将一个 zip 文件下载到 Notebooks 目录。我们需要解压并将图片移动到新目录 `stanford-dogs-dataset`。

## 数据集探索

Stanford Dogs 数据集包含 120 个犬种共 20580 张图片。数据集分为训练集和测试集，每个犬种训练集有 1000 张图片，测试集有 200 张。让我们看看其文件结构：



```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```


```python
from pathlib import Path

data_path = Path('stanford-dogs-dataset/images/Images')
# 每个目录是一个不同的犬种，如 "nn02107142-Doberman"
breeds = [d.name for d in data_path.iterdir() if d.is_dir()]
print(f"犬种数量: {len(breeds)}")
print(f"示例犬种: {breeds[:5]}")
```

    犬种数量: 120
    示例犬种: ['n02097658-silky_terrier', 'n02092002-Scottish_deerhound', 'n02099849-Chesapeake_Bay_retriever', 'n02091244-Ibizan_hound', 'n02095314-wire-haired_fox_terrier']



```python
# 显示某个犬种的 5 张随机图片
import matplotlib.pyplot as plt
from PIL import Image
import random

sample_breed = random.choice(breeds)
sample_images = list((data_path/sample_breed).glob('*.jpg'))[:5]

fig, axs = plt.subplots(1, 5, figsize=(15,3))
for ax, img_path in zip(axs, sample_images):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(sample_breed)
    ax.axis('off')
plt.show()
```


    
![fastai-part2-zlu-me](/assets/images/uploads/fastai/fastai-part2_3_0.png)
    


## fastai 数据准备

fastai 让加载和预处理图片数据变得很简单。我们将用 ImageDataLoaders.from_folder 方法，它要求图片按类别分文件夹存放。

关于变量命名。`item_tfms` 是 fastai 用语，表示对每张图片（item）应用的变换。个人认为 `tfms` 这个变量名不太直观，只取了 transform 的 4 个字母，每次用都要想怎么拼。直接用 item_transform 会更清楚，也不会因为多打几个字母得腱鞘炎。更让人困扰的是，这个变量其实可以更具体，比如 item_resize。严格来说，resize 也是 transform 的一种。fastai 为了统一，库里都用 `tfms`，但牺牲了明确性和易懂性。我并不推崇这种做法，更喜欢描述性强、易懂的变量名。当然这只是我的看法。

batch_tfms 参数用于对整个 batch 的图片做变换。建议对训练集和验证集都做同样的变换，这样可以防止过拟合。`aug_transforms` 函数会创建一组标准的数据增强变换，如随机翻转、旋转、缩放、扭曲等。mult=1.0 控制增强强度（1.0 为默认）。这就是机器学习（尤其是计算机视觉）中常说的数据增强，通过对图片做随机变换，人工增加训练集的多样性。
常见的数据增强包括：
- 水平或垂直翻转
- 小角度旋转
- 缩放
- 改变亮度或对比度
- 裁剪或扭曲


```python
from fastai.vision.all import *

dls = ImageDataLoaders.from_folder(
    data_path,
    valid_pct=0.2,        # 20% 用于验证集
    seed=42,
    item_tfms=Resize(224), # 图片缩放到 224x224
    batch_tfms=aug_transforms(mult=1.0) # 数据增强
)

dls.show_batch(max_n=9, figsize=(7,7))
```

    /opt/anaconda3/envs/ml_study/lib/python3.12/site-packages/torch/_tensor.py:1648: UserWarning: The operator 'aten::_linalg_solve_ex.result' is not currently supported on the MPS backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/mps/MPSFallback.mm:14.)
      ret = func(*args, **kwargs)



    
![fastai-zlu-me](/assets/images/uploads/fastai/fastai-part2_5_1.png)
    


### MPS 回退问题

在我的 M1 Macbook Pro 上运行上述代码会遇到一个已知问题：[PyTorch Issue #141287](https://github.com/pytorch/pytorch/issues/141287)。这是因为 PyTorch 目前还不完全支持 MPS 后端。临时解决办法是在导入其他库前设置环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让不支持的操作回退到 CPU。注意：这样会比直接用 MPS 慢。这也是为什么我们在 notebook 最开始就设置了 MPS fallback。

## 模型训练
我们将使用预训练的 ResNet34 模型，这是图像分类任务的良好起点。

fine_tune 方法会先训练模型的 head 部分，然后解冻并训练整个模型几轮。每个 epoch 后会显示训练和验证准确率。


```python
# 详见本系列第一部分的代码讲解
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /Users/zlu/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    100%|██████████| 83.3M/83.3M [00:06<00:00, 13.8MB/s]



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
      <td>1.518806</td>
      <td>0.669011</td>
      <td>0.792517</td>
      <td>03:30</td>
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
      <td>1.063292</td>
      <td>0.801603</td>
      <td>0.754616</td>
      <td>04:42</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.810561</td>
      <td>0.618079</td>
      <td>0.811710</td>
      <td>04:45</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.516495</td>
      <td>0.560497</td>
      <td>0.824830</td>
      <td>04:37</td>
    </tr>
  </tbody>
</table>


## 评估
让我们更详细地评估模型表现，并绘制混淆矩阵。
1. ClassificationInterpretation.from_learner(learn)：从训练好的 fastai Learner（这里叫 learn）创建一个 ClassificationInterpretation 对象。它提供分析和理解模型预测的工具，并收集模型预测、真实标签，计算哪些预测正确、哪些错误。

2. interp.plot_confusion_matrix(figsize=(12,12), dpi=60)：绘制模型预测的混淆矩阵。混淆矩阵是一张表，显示模型每个类别的预测与实际类别的对应情况。行表示实际类别，列表示预测类别。对角线（左上到右下）是预测正确的，非对角线是预测错误的。figsize 和 dpi 控制图像大小和分辨率。


```python
interp = ClassificationInterpretation.from_learner(learn)
# 实际上显示混淆矩阵很有意义，因为有 120 个犬种（类别）。
# 后面我们会处理这个问题。
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
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








    
![fastai-zlu-me](/assets/images/uploads/fastai/fastai-part2_10_6.png)
    



```python
most_confused = interp.most_confused()[:5]
print(most_confused)
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




<div>
  <progress value='0' class='' max='65' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.00% [0/65 00:00&lt;?]
</div>



    [('n02086240-Shih-Tzu', 'n02098413-Lhasa', np.int64(13)), ('n02110185-Siberian_husky', 'n02110063-malamute', np.int64(13)), ('n02093256-Staffordshire_bullterrier', 'n02093428-American_Staffordshire_terrier', np.int64(11)), ('n02106030-collie', 'n02106166-Border_collie', np.int64(11)), ('n02091032-Italian_greyhound', 'n02091134-whippet', np.int64(9))]



```python
# 展开混淆对，获取唯一类别名
confused_classes = set()
for a, b, _ in most_confused:
    confused_classes.add(a)
    confused_classes.add(b)
confused_classes = list(confused_classes)
print(confused_classes)

# 获取这些类别的索引
class2idx = {v: k for k, v in enumerate(interp.vocab)}
idxs = [class2idx[c] for c in confused_classes]
```

    ['n02110063-malamute', 'n02093256-Staffordshire_bullterrier', 'n02091134-whippet', 'n02093428-American_Staffordshire_terrier', 'n02091032-Italian_greyhound', 'n02086240-Shih-Tzu', 'n02110185-Siberian_husky', 'n02106166-Border_collie', 'n02106030-collie', 'n02098413-Lhasa']



```python
# 提取最易混淆类别的子混淆矩阵
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = interp.confusion_matrix()
sub_cm = cm[np.ix_(idxs, idxs)]

plt.figure(figsize=(8,6))
sns.heatmap(sub_cm, annot=True, fmt='d', 
            xticklabels=confused_classes, 
            yticklabels=confused_classes, cmap='Blues')
plt.xlabel('预测')
plt.ylabel('实际')
plt.title('最易混淆类别的混淆矩阵')
plt.show()
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








    
![fastai-zlu-me](/assets/images/uploads/fastai/fastai-part2_13_2.png)
    


### 如何读混淆矩阵

顾名思义，混淆矩阵的确容易让人困惑。

**在混淆矩阵中**

坐标轴：

行表示实际（真实）犬种标签。
列表示模型预测的犬种标签。

单元格：

每个单元格显示某个实际犬种（行）被预测为某个犬种（列）的图片数量。
对角线（左上到右下）是预测正确的。
非对角线是模型混淆的地方（预测错了）。

**上面的混淆矩阵显示：**

最易混淆的类别对：

这里展示的犬种是模型最容易混淆的。
比如 "Siberian_husky" 和 "malamute" 经常被混淆，这很合理，因为它们外观相似。
"Shih-Tzu" 和 "Lhasa" 也常被混淆，可能也是因为外观相似。

模型的强项与弱项：

模型总体上能区分这些犬种（对角线数值高）。
但有些类别对模型有困难，可能原因包括：
- 犬种外观相似
- 训练数据不足或标签模糊

### 查看 top losses：

这里的 loss 衡量模型对某张图片预测的错误程度。loss 越高，说明模型对错误预测越自信，或非常不确定。


```python
interp.plot_top_losses(9, nrows=3)
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








    
![fastai-zlu-me](/assets/images/uploads/fastai/fastai-part2_16_2.png)
    


## 模型推理
要在新图片上使用模型，只需调用 learn.predict：


```python
img = PILImage.create('stanford-dogs-dataset/images/Images/n02085620-Chihuahua/n02085620_10074.jpg')
pred_class, pred_idx, outputs = learn.predict(img)
print(f"预测类别: {pred_class}")
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







    预测类别: n02085620-Chihuahua


## 模型导出与导入
你也可以导出训练好的模型，之后再加载：


```python
learn.export('dog_breed_classifier.pkl')
# 重要：模型导出后再加载。
learn_inf = load_learner('dog_breed_classifier.pkl')
```

## 总结
在本篇中，我们用 fastai 库和 Stanford Dogs 数据集构建了一个犬种分类器。我们探索了数据集，准备了数据，训练了先进的深度学习模型，并评估了其表现。借助 fastai，只需几行代码就能实现高效原型开发，并在复杂的图像分类任务上获得高准确率。 