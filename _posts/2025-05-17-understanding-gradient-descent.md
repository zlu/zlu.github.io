---
layout: post
title: "Understanding Gradient Descent with a Sprinkle of Math"
date: 2025-05-17
last_modified_at: 2025-05-17
comments: true
categories: 
  - mathematics 
  - machine-learning 
  - optimization 
  - calculus 
  - programming
  - data-science
  - artificial-intelligence
tags:
  - gradient descent
  - optimization algorithms
  - machine learning basics
  - multivariable calculus
  - backpropagation
excerpt: "Master the fundamentals of gradient descent in machine learning with this comprehensive guide. Learn the mathematical foundations, from single-variable derivatives to multivariable gradients, with clear explanations and visual examples."
description: "A beginner-friendly yet comprehensive guide to understanding gradient descent in machine learning, covering the mathematical foundations from single-variable calculus to multivariable gradients, with clear explanations and visual examples."
image: /assets/images/uploads/gradients/3d_surface_gradient.png
---

![3D Surface Gradient
Descent](/assets/images/uploads/gradients/3d_surface_gradient.png)

Gradient Descent (GD) is probably one of the harder to understand concepts in the
early stage of the study of machine learning. The hill-climbing analogy, while
easy to understand, but leaves a feeling of sort-of understood, but not really.
It becomes more confusing when GD is mentioned in the context of loss function
together with backpropagation. I think to properly understand GD, it is
inevitable to learn a bit of its mathematical root. It is not much, but does
stir some curious folks away.

## The Foundation: Single-Variable Calculus

Gradient comes from multivariable calculus, which is an extension of
single-variable calculus. It can expressed as a funtion with a single variable $$f(x)$$. The derivative of $$f(x)$$, or $$f'(x)$$ (or $$\frac{df}{dx}$$), represents the rate of change of the function at a specific point $$x$$. Geometrically speaking, it is the slope of the tangent line to the function at that point.

For example, if $$f(x) = x^2$$, then $$f'(x) = 2x$$. At $$x = 5$$, the
derivative is $$f'(5) = 10$$, meaning that the function is increasing at a rate
of 10 units per unit change in $$x$$ at that point.

## Extending to Multiple Variables

By expanding the number of variables in the single-variable function, we get
multivariable function.  For example, a two-variable function is: $$f(x, y) =
x^2 + xy + y^2$$.  Then the partial derivative of $$f$$ with respect to $$x$$
can be denoted as: $$\frac{\partial f}{\partial x}$$, which measures the rate of
change of $$f$$ when only $$x$$ is varied.  Similarly, $$\frac{\partial
f}{\partial y}$$ measures the rate of change of $$f$$ when only $$y$$ varies.

The simple two-variable example above gives the follow two partial derivative:

$$\frac{\partial f}{\partial x} = 2x + y$$

$$\frac{\partial f}{\partial y} = x + 2y$$

With the vector notation, we can conveniently express the above function as:

$$\nabla f(x, y) = \left[\frac{\partial f}{\partial x}, \frac{\partial
f}{\partial y}\right]$$
Where $$\nabla f$$ is called "del f" or "grad f".

With vector notation, we can extend $$f$$ to any number of dimensions
(variables).

Suppose we have a three-dimensional surface representing the two-variable
function $$f(x, y)$$.  At any point $$(x, y)$$ on this surface, the gradient
$$\nabla f(x, y)$$ is a vector that:
- Points in the direction where the function increases most raplidly.
- Has a magnitude proportional to the rate of that increase.

If you drop a ball on the top part of this surface, it would let the gravity
taking it towards the direction of the bottom, or opposite to the direction of
the gradient.

## Application of GD in Optimization

When optimizing a loss function, we apply gradient descent by:
1. Start a some initial point.
2. Compute the gradient at that point.
3. Move in the direction opposition to the gradient by the following rule:
$$x_{t+1} = x_t - \eta \nabla f(x_t)$$, where $$\eta$$ (eta) is the learning
rate that determines the step sizze.
4. Repeat until convergence. 

The choice of learning rate $$\eta$$ is cruicial for the convergence of gradient descent.  If it is too small, then the convergence is slow (taking too much time in training the model).  If it is too large, then we may overshoot the minimum and fail to converge.

In neural networks such as backpropagation, we calculate the gradient of the
loss function with respect to each weight in the network by applying the chain
rule of calculus.

1. Forward pass: Go through each layer in the network, compute the output and
   loss.
2. Backward pass: Propagate the error backwards, compute gradients layer by
   layer.
3. Update weights using the computed gradients.

## Common Challenges in Gradient Descent

Several issues can arise when using gradients in deep learning:
1. Vanishing Gradients: Graidents become extrememly small as they propagate
   backward through layers, making learning in early layers very slow.
2. Exploding Gradients: Gradients become extremely large, causing upstable
   updates.
3. Saddle Points: Flat regions where gradients are close to zero but do not
   represent local minima.
4. Noisy Gradients: Especially in stochastic methods, gradients can be noisy
   esitmates of the true gradietn.

## Gradient Descent Variants

We may apply the following variants of gradient descent to overcome some of the
issues:
- Batch Gradient Descent: Compute the gradient using the entire dataset.
- Stochastic (random) Gradient Descent (SGD): Updates parameters using one
randomly selected sample at a time.
- Mini-batch Gradient Descent: Updates parameters using a small random batch of
samples.
- Adaptive Methods (AdaGrad, Adam): Adjust learning rates for each parameter
based on historical gradient information.

## Practical Example with TensorFlow

Now let's take a look and a code example in Tensorflow:
```python

import tensorflow as tf

# Define a function using TensorFlow operations
@tf.function
def f(x, y):
    return x**2 + x*y + y**2

# Point where we want to compute the gradient
x = tf.Variable(2.0)
y = tf.Variable(1.0)

# Use gradient tape to record operations for automatic differentiation
with tf.GradientTape() as tape:
    z = f(x, y)
    
# Compute gradient of z with respect to [x, y]
gradient = tape.gradient(z, [x, y])

print(f"Function value at (2, 1): {z.numpy()}")
print(f"Gradient: [df/dx, df/dy] = [{gradient[0].numpy()}, {gradient[1].numpy()}]")

# Implement gradient descent using TensorFlow
def tf_gradient_descent(start_point, learning_rate=0.1, iterations=100):
    x_value, y_value = start_point
    x = tf.Variable(float(x_value))
    y = tf.Variable(float(y_value))
    
    path = [(x.numpy(), y.numpy())]
    
    for i in range(iterations):
        with tf.GradientTape() as tape:
            z = f(x, y)
        
        # Get gradients
        [dx, dy] = tape.gradient(z, [x, y])
        
        # Update variables
        x.assign_sub(learning_rate * dx)
        y.assign_sub(learning_rate * dy)
        
        path.append((x.numpy(), y.numpy()))
        
        # Stop condition
        if tf.sqrt(dx**2 + dy**2) < 1e-6:
            break
            
    return path

# Run TensorFlow-based gradient descent
tf_path = tf_gradient_descent((3.0, 4.0))
print(f"TF Start point: {tf_path[0]}")
print(f"TF End point: {tf_path[-1]}")
print(f"TF Final function value: {f(tf.constant(tf_path[-1][0]), tf.constant(tf_path[-1][1])).numpy()}")
```

In this example:
1. We use GradientTape for automatic differentiation.
2. We compute gradients without manually deriving them.
3. We implement gradient descent with TensorFlow's automatic differentiation.

Numpy, PyTorch all provides convenient ways for GD.  We may touch base on them
in later blogs.

Thanks for reading and happy Saturday!
