---
layout: post
title: "分形艺术探秘：TypeScript 实现数学之美"
title_suffix: "zlu.me"
description: "学习如何使用 TypeScript 生成美丽的分形图像，了解分形背后的数学原理，探索其在艺术、金融等领域的实际应用。"
keywords: "分形, 曼德博集合, typescript, d3.js, 数学可视化, 分形艺术, 金融市场, 赫斯特指数"
date: 2025-05-18
comments: true
---

![fractal-mandel-brot](/assets/images/uploads/fractals-mandelbrot.png)

看更多: [algo-scope.online](https://www.algo-scape.online/fractals)

# 使用 TypeScript 生成分形图像

我一直被分形图像及其内在的图案和炫酷的色彩所吸引。在这些图像背后，隐藏着优雅的数学和递归函数。在这篇文章中，我们将学习什么是分形图像，如何生成它们，以及它们的实际应用。

## 历史

贝努瓦·B·曼德博在1975年创造了"分形"这个词。他用"分形"来描述那些无法用传统欧几里得几何定义，但可以用递归函数建模的形状。

## 如何生成分形图像

分形图像是通过应用递归变换的算法生成的。以曼德博集合为例：

$$ z\_{n+1} = z_n^2 + c $$

其中：

- 曼德博集合是所有满足由递推关系定义的序列的复数 $c$ 的集合。
- 从 $z_0 = 0$ 开始
- 继续计算 $ z_1 = z_0^2 + c $，然后是 $z_2 = z_1^2 + c$，以此类推
- 如果模长绝对值$z_n$保持有界（即不会趋向无穷大），则 $c$ 属于曼德博集合。
  数学上表示为：$ c \in \mathbb{C} \text{ 属于曼德博集合，如果 } \lim\_{n \to \infty} |z_n| < \infty $。

```typescript
private mandelbrot(x0: number, y0: number): number {
  let x = 0;  // z 的实部
  let y = 0;  // z 的虚部
  let iteration = 0;

  while (x * x + y * y <= 4 && iteration < this.maxIterations) {
    const xTemp = x * x - y * y + x0;  // z² + c 的实部
    y = 2 * x * y + y0;                // z² + c 的虚部
    x = xTemp;
    iteration++;
  }

  return iteration;
}
```

## 渲染分形

与 [Algo-Scope.online](https://algo-scope.online) 上的其他可视化一样，我们使用 d3.js 进行渲染，以获得更好的控制和缩放功能。这是一个示例：

```typescript
function renderFractal(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  data: FractalPoint[],
  maxIterations: number,
) {
  const colorScale = d3
    .scaleSequential(d3.interpolateViridis)
    .domain([0, maxIterations]);

  svg
    .selectAll("rect")
    .data(data)
    .enter()
    .append("rect")
    .attr("x", (d) => d.x)
    .attr("y", (d) => d.y)
    .attr("width", 1)
    .attr("height", 1)
    .attr("fill", (d) => colorScale(d.value));
}
```

更多示例（如朱利亚集、牛顿分形等）可以在 algo-scope 上找到供您实验。

## 分形的应用

分形不仅仅是数学上的奇观，它们在艺术设计、计算机图形学、医学、信号处理，以及金融等多个领域都有广泛的应用。

在艺术领域，分形生成艺术使用算法创造出视觉上令人惊叹的复杂作品。

![3-D Fractal by Babymilk](/assets/images/uploads/babymilk-fractal.png)
[查看原图](https://www.deviantart.com/babymik/art/uhm-63065658)

在金融领域，分形经常被用来模拟市场波动，因为它们具有不规则但自相似的时间行为。正如曼德博敏锐地指出，金融市场表现出"分形性"，波动模式在不同时间尺度上重复出现——从秒到年。

金融市场假说（FMH）是一个理论，认为市场受到不同时间跨度的投资者集体行为的影响。赫斯特指数和多分形模型被用来检测持续性趋势或波动聚集，这些是线性模型无法捕捉的。

![fractal-hurst-exponent-zlu-me-algo-scape-online](/assets/images/uploads/fractal-hurst-exp.png)

## 结论

这是对分形图像的介绍，我们了解了其背后的数学原理和生成方法。在后续系列中，我们将探讨一些更高级的例子，如牛顿分形。我们还将深入研究如何使用 TypeScript 在网络上正确部署分形并使其具有交互性。这种方法有一些性能影响。我们如何利用 Web Workers 进行并行处理，如何渐进式渲染复杂图像，如何应用自适应迭代限制，当然还有缓存和记忆化。

分形的美本质上是递归的。它连接了数学和视觉艺术，值得我们关注。

感谢阅读，祝周日愉快！ 