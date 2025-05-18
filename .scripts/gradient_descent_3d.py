import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Define the function to optimize (example: f(x) = x^2)
def f(x):
    return x**2

# Define the gradient of the function
def gradient(x):
    return 2*x

# Gradient descent parameters
learning_rate = 0.1
n_iterations = 10
initial_point = 2.0

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create points for the function curve
x = np.linspace(-2.5, 2.5, 100)
y = f(x)

# Plot the function curve
ax.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')

# Initialize the point and path
point = initial_point
path = [point]
gradients = []

# Perform gradient descent
for _ in range(n_iterations):
    grad = gradient(point)
    point = point - learning_rate * grad
    path.append(point)
    gradients.append(grad)

path = np.array(path)
gradients = np.array(gradients)

# Plot the path points
ax.scatter(path, f(path), c='red', s=100, label='Steps')

# Plot gradient arrows at each point
for i in range(len(path)-1):
    current_point = path[i]
    current_grad = gradients[i]
    
    # Calculate the arrow components
    # The gradient gives us the slope, so we use it to calculate dx and dy
    dx = -current_grad * 0.3  # Negative because we want to show descent direction
    dy = -current_grad * dx   # This gives us the vertical component
    
    # Plot the arrow
    ax.arrow(current_point, f(current_point),
             dx, dy,
             head_width=0.1, head_length=0.1,
             fc='red', ec='red',
             alpha=0.6,
             label=f'Gradient at step {i}' if i == 0 else "")

# Add a point at the minimum
ax.scatter([0], [0], c='green', s=100, label='Minimum', zorder=5)

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('1D Gradient Descent Visualization')
ax.legend()

# Set axis limits
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-0.5, 5])

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show() 