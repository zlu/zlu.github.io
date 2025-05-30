在fastai第一部分中，我们学习了如何对MNIST数据集进行分类。在本教程中，我们将深入探讨，了解其背后的工作原理。首先，我们将深入了解MNIST数据集。

## 数据探索


```python
# 来自第一部分的代码
import torch
import random
from fastai.vision.all import *

# 下载简单的MNIST数据集（还不是完整数据集，我们稍后会下载）
path = untar_data(URLs.MNIST_SAMPLE)
train_path = path/'train'
img_files = list((train_path/'7').ls())
img = PILImage.create(img_files[0])
img.show();
```


    
![png](fastai-part3_files/fastai-part3_1_0.png)
    



```python
print(f"图像模式为: {img.mode}")
```

    图像模式为: RGB


MNIST图像是灰度图（单通道）。当通过PILImage创建时，它们被转换为RGB格式，这可能是为了模型兼容性。这就是为什么如上所示的打印输出中，图像的模式现在是RGB。灰度图像会显示为"L"（1个通道），而不是RGB（3个通道）。如果你想保持图像为灰度，可以使用以下代码：
```python
img = PILImage.create(img_files[0])
img.show();
print(f"图像模式为: {img.mode}")
```


```python
arr = array(img)
print(arr.shape)
```

    (28, 28, 3)


(28, 28, 3)是一个三维NumPy数组，表示高度、宽度和通道。它的大小为28x28像素。'3'表示RGB通道，由于图像是灰度的，这些通道的值可能是相同的。


```python
print(np.unique(arr))
```

    [  0   9  23  24  34  38  44  46  47  64  69  71  76  93  99 104 107 109
     111 115 128 137 138 139 145 146 149 151 154 161 168 174 176 180 184 185
     207 208 214 215 221 230 231 240 244 245 251 253 254 255]


`np.unique(arr)`给出了一系列值，显示这些像素值的范围从0到255。'0'是黑色，'255'是白色。中间的值是灰色的不同色调。


```python
np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2])
```




    np.True_



上面的代码检查所有三个通道的像素值是否相同。由于图像是灰度的，值是相同的，因此结果为np.True。


```python
img_t = tensor(arr[:, :, 0])
print(img_t.shape)
df = pd.DataFrame(img_t)
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')
```

    torch.Size([28, 28])



<style type="text/css">
#T_8a8f1_row0_col0, #T_8a8f1_row0_col1, #T_8a8f1_row0_col2, #T_8a8f1_row0_col3, #T_8a8f1_row0_col4, #T_8a8f1_row0_col5, #T_8a8f1_row0_col6, #T_8a8f1_row0_col7, #T_8a8f1_row0_col8, #T_8a8f1_row0_col9, #T_8a8f1_row0_col10, #T_8a8f1_row0_col11, #T_8a8f1_row0_col12, #T_8a8f1_row0_col13, #T_8a8f1_row0_col14, #T_8a8f1_row0_col15, #T_8a8f1_row0_col16, #T_8a8f1_row0_col17, #T_8a8f1_row0_col18, #T_8a8f1_row0_col19, #T_8a8f1_row0_col20, #T_8a8f1_row0_col21, #T_8a8f1_row0_col22, #T_8a8f1_row0_col23, #T_8a8f1_row0_col24, #T_8a8f1_row0_col25, #T_8a8f1_row0_col26, #T_8a8f1_row0_col27, #T_8a8f1_row1_col0, #T_8a8f1_row1_col1, #T_8a8f1_row1_col2, #T_8a8f1_row1_col3, #T_8a8f1_row1_col4, #T_8a8f1_row1_col5, #T_8a8f1_row1_col6, #T_8a8f1_row1_col7, #T_8a8f1_row1_col8, #T_8a8f1_row1_col9, #T_8a8f1_row1_col10, #T_8a8f1_row1_col11, #T_8a8f1_row1_col12, #T_8a8f1_row1_col13, #T_8a8f1_row1_col14, #T_8a8f1_row1_col15, #T_8a8f1_row1_col16, #T_8a8f1_row1_col17, #T_8a8f1_row1_col18, #T_8a8f1_row1_col19, #T_8a8f1_row1_col20, #T_8a8f1_row1_col21, #T_8a8f1_row1_col22, #T_8a8f1_row1_col23, #T_8a8f1_row1_col24, #T_8a8f1_row1_col25, #T_8a8f1_row1_col26, #T_8a8f1_row1_col27, #T_8a8f1_row2_col0, #T_8a8f1_row2_col1, #T_8a8f1_row2_col2, #T_8a8f1_row2_col3, #T_8a8f1_row2_col4, #T_8a8f1_row2_col5, #T_8a8f1_row2_col6, #T_8a8f1_row2_col7, #T_8a8f1_row2_col8, #T_8a8f1_row2_col9, #T_8a8f1_row2_col10, #T_8a8f1_row2_col11, #T_8a8f1_row2_col12, #T_8a8f1_row2_col13, #T_8a8f1_row2_col14, #T_8a8f1_row2_col15, #T_8a8f1_row2_col16, #T_8a8f1_row2_col17, #T_8a8f1_row2_col18, #T_8a8f1_row2_col19, #T_8a8f1_row2_col20, #T_8a8f1_row2_col21, #T_8a8f1_row2_col22, #T_8a8f1_row2_col23, #T_8a8f1_row2_col24, #T_8a8f1_row2_col25, #T_8a8f1_row2_col26, #T_8a8f1_row2_col27, #T_8a8f1_row3_col0, #T_8a8f1_row3_col1, #T_8a8f1_row3_col2, #T_8a8f1_row3_col3, #T_8a8f1_row3_col4, #T_8a8f1_row3_col5, #T_8a8f1_row3_col6, #T_8a8f1_row3_col7, #T_8a8f1_row3_col8, #T_8a8f1_row3_col9, #T_8a8f1_row3_col10, #T_8a8f1_row3_col11, #T_8a8f1_row3_col12, #T_8a8f1_row3_col13, #T_8a8f1_row3_col14, #T_8a8f1_row3_col15, #T_8a8f1_row3_col16, #T_8a8f1_row3_col17, #T_8a8f1_row3_col18, #T_8a8f1_row3_col19, #T_8a8f1_row3_col20, #T_8a8f1_row3_col21, #T_8a8f1_row3_col22, #T_8a8f1_row3_col23, #T_8a8f1_row3_col24, #T_8a8f1_row3_col25, #T_8a8f1_row3_col26, #T_8a8f1_row3_col27, #T_8a8f1_row4_col0, #T_8a8f1_row4_col1, #T_8a8f1_row4_col2, #T_8a8f1_row4_col3, #T_8a8f1_row4_col4, #T_8a8f1_row4_col5, #T_8a8f1_row4_col6, #T_8a8f1_row4_col7, #T_8a8f1_row4_col8, #T_8a8f1_row4_col9, #T_8a8f1_row4_col10, #T_8a8f1_row4_col11, #T_8a8f1_row4_col12, #T_8a8f1_row4_col13, #T_8a8f1_row4_col14, #T_8a8f1_row4_col15, #T_8a8f1_row4_col16, #T_8a8f1_row4_col17, #T_8a8f1_row4_col18, #T_8a8f1_row4_col19, #T_8a8f1_row4_col20, #T_8a8f1_row4_col21, #T_8a8f1_row4_col22, #T_8a8f1_row4_col23, #T_8a8f1_row4_col24, #T_8a8f1_row4_col25, #T_8a8f1_row4_col26, #T_8a8f1_row4_col27, #T_8a8f1_row5_col0, #T_8a8f1_row5_col1, #T_8a8f1_row5_col2, #T_8a8f1_row5_col3, #T_8a8f1_row5_col4, #T_8a8f1_row5_col5, #T_8a8f1_row5_col6, #T_8a8f1_row5_col7, #T_8a8f1_row5_col8, #T_8a8f1_row5_col9, #T_8a8f1_row5_col10, #T_8a8f1_row5_col11, #T_8a8f1_row5_col12, #T_8a8f1_row5_col13, #T_8a8f1_row5_col14, #T_8a8f1_row5_col15, #T_8a8f1_row5_col16, #T_8a8f1_row5_col17, #T_8a8f1_row5_col18, #T_8a8f1_row5_col19, #T_8a8f1_row5_col20, #T_8a8f1_row5_col21, #T_8a8f1_row5_col22, #T_8a8f1_row5_col23, #T_8a8f1_row5_col24, #T_8a8f1_row5_col25, #T_8a8f1_row5_col26, #T_8a8f1_row5_col27, #T_8a8f1_row6_col0, #T_8a8f1_row6_col1, #T_8a8f1_row6_col2, #T_8a8f1_row6_col3, #T_8a8f1_row6_col4, #T_8a8f1_row6_col5, #T_8a8f1_row6_col6, #T_8a8f1_row6_col7, #T_8a8f1_row6_col8, #T_8a8f1_row6_col9, #T_8a8f1_row6_col10, #T_8a8f1_row6_col11, #T_8a8f1_row6_col12, #T_8a8f1_row6_col13, #T_8a8f1_row6_col14, #T_8a8f1_row6_col15, #T_8a8f1_row6_col16, #T_8a8f1_row6_col17, #T_8a8f1_row6_col18, #T_8a8f1_row6_col19, #T_8a8f1_row6_col20, #T_8a8f1_row6_col21, #T_8a8f1_row6_col22, #T_8a8f1_row6_col23, #T_8a8f1_row6_col24, #T_8a8f1_row6_col25, #T_8a8f1_row6_col26, #T_8a8f1_row6_col27, #T_8a8f1_row7_col0, #T_8a8f1_row7_col1, #T_8a8f1_row7_col2, #T_8a8f1_row7_col3, #T_8a8f1_row7_col4, #T_8a8f1_row7_col5, #T_8a8f1_row7_col6, #T_8a8f1_row7_col7, #T_8a8f1_row7_col8, #T_8a8f1_row7_col9, #T_8a8f1_row7_col10, #T_8a8f1_row7_col11, #T_8a8f1_row7_col12, #T_8a8f1_row7_col13, #T_8a8f1_row7_col20, #T_8a8f1_row7_col21, #T_8a8f1_row7_col22, #T_8a8f1_row7_col23, #T_8a8f1_row7_col24, #T_8a8f1_row7_col25, #T_8a8f1_row7_col26, #T_8a8f1_row7_col27, #T_8a8f1_row8_col0, #T_8a8f1_row8_col1, #T_8a8f1_row8_col20, #T_8a8f1_row8_col21, #T_8a8f1_row8_col22, #T_8a8f1_row8_col23, #T_8a8f1_row8_col24, #T_8a8f1_row8_col25, #T_8a8f1_row8_col26, #T_8a8f1_row8_col27, #T_8a8f1_row9_col0, #T_8a8f1_row9_col1, #T_8a8f1_row9_col20, #T_8a8f1_row9_col21, #T_8a8f1_row9_col22, #T_8a8f1_row9_col23, #T_8a8f1_row9_col24, #T_8a8f1_row9_col25, #T_8a8f1_row9_col26, #T_8a8f1_row9_col27, #T_8a8f1_row10_col0, #T_8a8f1_row10_col1, #T_8a8f1_row10_col2, #T_8a8f1_row10_col3, #T_8a8f1_row10_col4, #T_8a8f1_row10_col13, #T_8a8f1_row10_col14, #T_8a8f1_row10_col15, #T_8a8f1_row10_col16, #T_8a8f1_row10_col20, #T_8a8f1_row10_col21, #T_8a8f1_row10_col22, #T_8a8f1_row10_col23, #T_8a8f1_row10_col24, #T_8a8f1_row10_col25, #T_8a8f1_row10_col26, #T_8a8f1_row10_col27 {
  font-size: 6pt;
  background-color: #ffffff;
  color: #000000;
}
</style>

上面的代码将图像转换为张量，并使用pandas DataFrame和样式功能以灰度渐变显示图像。这提供了数字"7"的可视化表示。

## 基准图像

让我们看看每个数字类别的一些示例图像：


```python
# 获取每个数字类别的一个示例
sample_imgs = {}
for i in range(10):
    sample_files = list((train_path/f'{i}').ls())
    if sample_files:  # 确保文件夹不为空
        sample_imgs[i] = PILImage.create(sample_files[0])

# 显示每个数字的示例
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i in sample_imgs:
        ax.imshow(sample_imgs[i], cmap='gray')
        ax.set_title(f"数字: {i}")
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
```

## 在完整的MNIST数据集上训练神经网络

现在，让我们下载完整的MNIST数据集并训练一个神经网络：


```python
# 下载完整的MNIST数据集
path = untar_data(URLs.MNIST)
path.ls()
```

### DataBlock

我们将使用fastai的DataBlock API来准备数据：


```python
# 定义数据块
mnist = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='training', valid_name='testing'),
    get_y=parent_label
)

# 创建数据加载器
dls = mnist.dataloaders(path, bs=128)

# 查看一批数据
dls.show_batch(max_n=9, figsize=(6, 6))
```

现在，让我们创建并训练一个简单的卷积神经网络：


```python
# 定义一个简单的CNN模型
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# 创建学习器
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

# 训练模型
learn.fit_one_cycle(5, 1e-3)
```

## 模型评估

让我们评估我们的模型性能：


```python
# 混淆矩阵
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(8, 8))
```

```python
# 查看最大损失的示例
interp.plot_top_losses(9, figsize=(7, 7))
```

## 进行预测

现在，让我们使用我们的模型进行一些预测：


```python
# 从测试集获取一些图像
test_files = get_image_files(path/'testing')
test_dl = learn.dls.test_dl(test_files[:10])

# 进行预测
preds, _ = learn.get_preds(dl=test_dl)
pred_classes = preds.argmax(dim=1)

# 显示图像和预测
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i, (img_file, pred) in enumerate(zip(test_files[:10], pred_classes)):
    img = PILImage.create(img_file)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"预测: {pred.item()}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

## 与我们的模板匹配方法比较

让我们比较我们的神经网络与之前的模板匹配方法：


```python
# 加载一些测试图像
test_imgs = [PILImage.create(f) for f in test_files[:5]]

# 显示结果比较
fig, axes = plt.subplots(5, 3, figsize=(12, 15))

for i, img in enumerate(test_imgs):
    # 原始图像
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title("原始图像")
    axes[i, 0].axis('off')
    
    # 模板匹配结果（假设）
    axes[i, 1].imshow(img, cmap='gray')
    axes[i, 1].set_title("模板匹配")
    axes[i, 1].axis('off')
    
    # 神经网络预测
    pred = learn.predict(img)[0]
    axes[i, 2].imshow(img, cmap='gray')
    axes[i, 2].set_title(f"神经网络: {pred}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()
```

## 可视化特征图

让我们可视化我们的CNN的第一个卷积层学到的特征：


```python
# 获取第一个卷积层
conv1 = model[0]

# 获取一批图像
x, y = dls.one_batch()

# 应用第一个卷积层获取激活值
with torch.no_grad():
    activations = conv1(x)

# 可视化第一张图像的激活值
# 我们的自定义模型在第一层有16个过滤器
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# 显示原始图像
axes[0].imshow(x[0][0].cpu(), cmap='gray')
axes[0].set_title(f"原始图像: {y[0].item()}")
axes[0].axis('off')

# 显示前15个过滤器的激活图
for i in range(1, 16):
    axes[i].imshow(activations[0, i-1].detach().cpu(), cmap='viridis')
    axes[i].set_title(f"过滤器 {i}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 同时可视化过滤器权重
weights = conv1.weight.data.cpu()
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

# 再次显示原始图像
axes[0].imshow(x[0][0].cpu(), cmap='gray')
axes[0].set_title(f"原始图像: {y[0].item()}")
axes[0].axis('off')

# 显示前15个过滤器的权重
for i in range(1, 16):
    # 每个过滤器只有一个输入通道（灰度）
    axes[i].imshow(weights[i-1, 0], cmap='viridis')
    axes[i].set_title(f"过滤器 {i} 权重")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

## 结论

在本教程中，我们深入研究了MNIST数据集，并使用fastai和PyTorch构建了一个简单的卷积神经网络来对手写数字进行分类。我们探索了数据，训练了模型，评估了其性能，并可视化了学习到的特征。

与我们之前的模板匹配方法相比，神经网络提供了更高的准确性和更好的泛化能力。通过可视化卷积层的激活和权重，我们可以了解网络如何学习识别不同的数字特征。

这种深度学习方法在处理更复杂的图像识别任务时特别有价值，因为它可以自动学习相关特征，而不需要手动设计特征提取器。