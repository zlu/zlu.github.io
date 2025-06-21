---
layout: post
title: 使用 R 处理图像
title_cn: true
date: 2025-06-21 09:00 +0800
tags:
  - R
  - 图像处理
lang_en: |
  Image processing in R, using packages like `imager`, enables powerful
  analysis and manipulation of digital images. This blog demonstrates key
  techniques—enhancement, denoising, and histogram equalization—using the
  `imager` package, with visual results to illustrate their effects, based
  on concepts from the "Image Data Analysis" document.
lang_cn: |
  在 R 中进行图像处理，使用像 `imager` 这样的包，可以实现强大的数字图像分析和处理。本博客将基于"图像数据分析"文档的概念，演示使用 `imager` 包进行的关键技术——图像增强、去噪和直方图均衡化，并通过可视化结果展示这些效果。

## 理解数字图像

数字图像是一个二维函数 $f(x, y): \mathbb{R}^2 \rightarrow \mathbb{R}$，其中 $(x, y)$ 是空间坐标，$f(x, y)$ 表示像素强度。在数字图像中，像素具有离散的位置和值。对于8位灰度图像，强度范围从0（黑色）到255（白色）。图像可分为以下类别：

- **二值图像**：像素值为0（黑色）或1（白色）
- **灰度图像**：像素值在 $\{0, \ldots, 255\}$ 范围内
- **彩色图像**：三个通道（RGB），每个通道的值在 $\{0, \ldots, 255\}$ 范围内

## 使用 `imager` 处理图像

`imager` 包基于CImg构建，支持图像的加载、处理和可视化。下面，我们将使用内置的"parrots.png"图像，通过代码和可视化输出来演示关键操作。

### 1. 图像获取

加载并显示图像：

```r
library(imager)
file <- system.file('extdata/parrots.png', package='imager')
img <- load.image(file)
plot(img, main="原始鹦鹉图像")
```

**结果**：
![原始鹦鹉图像](/assets/images/uploads/plots/original_parrots.png)

### 2. 图像增强（模糊）

模糊处理可以增强特定应用中的图像，例如平滑细节：

```r
img_blurry <- isoblur(img, sigma=10)
plot(img_blurry, main="模糊图像 (sigma=10)")
```

**结果**：
![模糊鹦鹉图像](/assets/images/uploads/plots/blurred_parrots.png)
_注意：模糊后的图像更加平滑，减少了羽毛纹理等细节。_

### 3. 图像去噪

去噪可以在保持结构的同时去除噪声。添加噪声并应用各向异性模糊：

```r
img_noisy <- img + 0.5 * rnorm(prod(dim(img)))
img_denoised <- blur_anisotropic(img_noisy, ampl=1e3, sharp=0.3)
layout(t(1:2))
plot(img_noisy, main="含噪图像")
plot(img_denoised, main="去噪后图像（各向异性）")
```

**结果**：
![含噪与去噪对比](/assets/images/uploads/plots/noisy_vs_denoised_parrots.png)
_注意：含噪图像显示随机斑点，而去噪后的图像恢复了清晰度，同时保持了边缘特征。_

### 4. 直方图均衡化

直方图均衡化通过重新分配像素强度来增强对比度：

```r
img_gray <- grayscale(img)
f <- ecdf(img_gray)
img_equalized <- f(img_gray) %>% as.cimg(dim=dim(img_gray))
layout(t(1:2))
plot(img_gray, main="灰度图像")
plot(img_equalized, main="直方图均衡化后图像")
```

**结果**：
![灰度与均衡化对比](/assets/images/uploads/plots/grayscale_vs_equalized_parrots.png)
_注意：均衡化后的图像对比度得到改善，使得颜色变化等细节更加明显。_

### 5. 形态学处理

通过强度进行阈值分割对象：

```r
img_gray <- grayscale(img)
threshold(img_gray, "20%") %>% plot(main="阈值处理后图像 (20%)")
```

**结果**：
![阈值处理后图像](/assets/images/uploads/plots/thresholded_parrots.png)
_注意：阈值处理创建了一个二值图像，突出显示了较亮区域（如白色羽毛）与较暗区域的对比。_

### 6. 完整代码

```r
library(imager)

# 设置保存图像的目录（可选：如果不存在则创建）
output_dir <- "plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 图像获取
file <- system.file('extdata/parrots.png', package='imager')
img <- load.image(file)
plot(img, main="原始鹦鹉图像")
dev.copy(png, file.path(output_dir, "original_parrots.png"))
dev.off()

# 图像增强（模糊）
img_blurry <- isoblur(img, sigma=10)
plot(img_blurry, main="模糊图像 (sigma=10)")
dev.copy(png, file.path(output_dir, "blurred_parrots.png"))
dev.off()

# 图像去噪
img_noisy <- img + 0.5 * rnorm(prod(dim(img)))
img_denoised <- blur_anisotropic(img_noisy, ampl=1e3, sharp=0.3)
layout(t(1:2))
plot(img_noisy, main="含噪图像")
plot(img_denoised, main="去噪后图像（各向异性）")
dev.copy(png, file.path(output_dir, "noisy_vs_denoised_parrots.png"))
dev.off()

# 直方图均衡化
img_gray <- grayscale(img)
f <- ecdf(img_gray)
img_equalized <- f(img_gray) %>% as.cimg(dim=dim(img_gray))
layout(t(1:2))
plot(img_gray, main="灰度图像")
plot(img_equalized, main="直方图均衡化后图像")
dev.copy(png, file.path(output_dir, "grayscale_vs_equalized_parrots.png"))
dev.off()

# 形态学处理
img_gray <- grayscale(img)
threshold(img_gray, "20%") %>% plot(main="阈值处理后图像 (20%)")
dev.copy(png, file.path(output_dir, "thresholded_parrots.png"))
dev.off()
```

## 应用场景

R中的图像处理应用于以下领域：
- **汽车工业**：车道检测、障碍物警告
- **医疗**：诊断成像、手术辅助
- **安防**：人脸识别、监控
- **媒体**：特效、图像编辑

## 结论

R中的 `imager` 包简化了图像处理任务，如增强、去噪和直方图均衡化。可视化结果展示了这些技术如何转换图像，改善质量或提取特征。探索 `imager` 和 `imagerExtra` 以进行更高级的应用。

**资源**：

- [imager包文档](https://cran.r-project.org/web/packages/imager/imager.pdf)
- [imager入门指南](https://cran.r-project.org/web/packages/imager/vignettes/gettingstarted.html)
- [imagerExtra指南](https://cran.r-project.org/web/packages/imagerExtra/vignettes/gettingstarted.html)
