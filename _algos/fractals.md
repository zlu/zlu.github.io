---
title: "Fractal generator"
description: "Interactive fractal explorations on AlgoScope: pan and recolor fractal families while tying escape-time math to the visual patterns you see on screen."
keywords: "fractal generator, Mandelbrot set, Julia set, TypeScript fractal, mathematical visualization"
demo: "https://algoscape.zlu.me/fractals"
order: 40
related_posts:
  - "_posts/2025-05-18-fractal.md"
  - "_posts/2025-05-18-fractal-cn.md"
---

Fractals are self-similar structures produced by simple iterative rules. The Mandelbrot set, for example, classifies each complex number $c$ by whether the recurrence $z_{n+1} = z_n^2 + c$ stays bounded. Escape-time coloring turns that dynamics into the familiar images: how quickly $|z|$ diverges maps to hue or brightness.

## Why it belongs here

The generator is an algorithm visualization as much as an art toy — nested loops over the complex plane, a stopping criterion, and a color map. The longer write-up, [Fractals Unveiled](/2025/05/18/fractal.html), also covers Hurst exponents and where fractal ideas show up outside pure math.

## Try it

Open the [fractal generator on AlgoScape](https://algoscape.zlu.me/fractals) and pan or recolor different fractal families.
