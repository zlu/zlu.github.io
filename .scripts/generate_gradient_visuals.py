#!/usr/bin/env python3
"""
Generate Gradient Visualizations for Blog Post
This script generates various gradient visualizations and saves them to the specified directory.
"""

import os
import sys

# Check dependencies
REQUIRED_PACKAGES = ["numpy", "matplotlib"]
MISSING_PACKAGES = []

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        MISSING_PACKAGES.append(package)

if MISSING_PACKAGES:
    print(f"Error: Missing required packages: {', '.join(MISSING_PACKAGES)}")
    print("Please install them using:")
    print(f"pip install {' '.join(MISSING_PACKAGES)}")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Warning: 3D plotting might not work properly. Please ensure mpl_toolkits is installed.")

# Create directory for saving visualizations
SAVE_DIR = "assets/images/uploads/gradients"
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Output directory: {SAVE_DIR}")
except Exception as e:
    print(f"Error creating directory: {e}")
    sys.exit(1)

# Define our example function and its gradient
def f(x, y):
    """The function f(x,y) = x^2 + xy + y^2"""
    return x**2 + x*y + y**2

def grad_f(x, y):
    """The gradient of f(x,y) = x^2 + xy + y^2"""
    return np.array([2*x + y, x + 2*y])

# Set global matplotlib parameters for better web display and smaller file sizes
plt.rcParams.update({
    'font.size': 9,            # Base font size
    'axes.labelsize': 9,       # Size of axis labels
    'axes.titlesize': 11,      # Size of plot title
    'xtick.labelsize': 8,      # Size of x-tick labels
    'ytick.labelsize': 8,      # Size of y-tick labels
    'legend.fontsize': 8,      # Size of legend text
    'figure.titlesize': 11,    # Size of figure title
    'figure.dpi': 100,         # Default DPI for saving
    'savefig.dpi': 120,        # Lower DPI for web images
    'savefig.bbox': 'tight',   # Tight bounding box 
    'savefig.pad_inches': 0.2, # Small padding
    'figure.figsize': (5, 4),  # Default figure size
    'figure.autolayout': True, # Better layout
})

# Function to handle exceptions during plotting
def safe_plot(plot_function, filename):
    """Execute a plotting function with error handling"""
    try:
        # Using a smaller figure size for web display
        plt.figure(figsize=(5, 4))
        plot_function()
        # Lower DPI for web-friendly file sizes
        plt.savefig(os.path.join(SAVE_DIR, filename), dpi=120, 
                   bbox_inches='tight', pad_inches=0.2,
                   transparent=False)
        plt.close()
        print(f"Successfully generated: {filename}")
        return True
    except Exception as e:
        print(f"Error generating {filename}: {e}")
        return False

def generate_all_visualizations():
    """Generate all visualizations for the blog post"""
    successful = 0
    total = 4  # Total number of visualizations

    # 1. Gradient Field Visualization
    def plot_gradient_field():
        # Create a grid of points
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)

        # Calculate function values on the grid
        Z = f(X, Y)

        # Calculate gradient components on the grid
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                U[i, j], V[i, j] = grad_f(X[i, j], Y[i, j])

        # Normalize the gradient vectors for better visualization
        norm = np.sqrt(U**2 + V**2)
        U_norm = U / norm
        V_norm = V / norm

        # Plot contour lines
        contour = plt.contour(X, Y, Z, 20, cmap=cm.viridis)
        plt.clabel(contour, inline=True, fontsize=8)

        # Plot gradient field
        plt.quiver(X, Y, U_norm, V_norm, norm, cmap=cm.autumn, scale=30, width=0.002)

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Gradient Magnitude')

        # Add labels and title
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Gradient Field for $f(x,y) = x^2 + xy + y^2$')
        plt.grid(True)

    # 2. Gradient Descent Paths Visualization
    def plot_gradient_descent_paths():
        def gradient_descent_path(start_point, learning_rate=0.1, iterations=30):
            """Compute gradient descent path from start_point"""
            path = [np.array(start_point, dtype=float)]
            point = path[0].copy()
            
            for i in range(iterations):
                grad = grad_f(point[0], point[1])
                point = point - learning_rate * grad
                path.append(point.copy())
                
                # Stop if gradient becomes very small
                if np.linalg.norm(grad) < 1e-6:
                    break
                    
            return np.array(path)

        # Plot contour lines
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        contour = plt.contour(X, Y, Z, 20, cmap=cm.viridis)
        plt.clabel(contour, inline=True, fontsize=8)

        # Compute and plot gradient descent paths from different starting points
        start_points = [(-2, 2), (3, 1), (-1, -2), (2, -2)]
        colors = ['red', 'blue', 'green', 'purple']

        for i, start in enumerate(start_points):
            path = gradient_descent_path(start, learning_rate=0.1)
            plt.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], 
                    label=f'Start: {start}', markersize=4)
            plt.annotate(f'Start {i+1}', start, fontsize=10, 
                        xytext=(10, 10), textcoords='offset points')

        # Mark the minimum at (0, 0)
        plt.plot(0, 0, 'r*', markersize=10, label='Minimum')
        plt.annotate('Minimum (0, 0)', (0, 0), fontsize=10, 
                    xytext=(10, -20), textcoords='offset points')

        # Add labels and title
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Gradient Descent Paths for $f(x,y) = x^2 + xy + y^2$')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    # 3. 3D Surface with Gradient Visualization
    def plot_3d_surface():
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(111, projection='3d')

        # Create a finer grid for the surface
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        # Plot the surface
        surface = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                                linewidth=0, antialiased=True)

        # Create a coarser grid for the gradient vectors
        x_arrow = np.linspace(-2, 2, 8)
        y_arrow = np.linspace(-2, 2, 8)
        X_arrow, Y_arrow = np.meshgrid(x_arrow, y_arrow)
        Z_arrow = f(X_arrow, Y_arrow)

        # Calculate gradient at each point
        U, V = np.zeros_like(X_arrow), np.zeros_like(Y_arrow)
        for i in range(X_arrow.shape[0]):
            for j in range(X_arrow.shape[1]):
                U[i, j], V[i, j] = grad_f(X_arrow[i, j], Y_arrow[i, j])

        # Normalize for consistent arrow length
        W = np.ones_like(Z_arrow) * 0.1  # Set a consistent z-component for visualization
        norm = np.sqrt(U**2 + V**2 + W**2)
        scale = 0.3  # Scale factor for arrow size
        U = scale * U / norm
        V = scale * V / norm
        W = scale * W / norm

        # Plot gradient vectors as 3D arrows
        ax.quiver(X_arrow, Y_arrow, Z_arrow, U, V, W, color='red', length=0.5, 
                normalize=False, arrow_length_ratio=0.3)

        # Add a point to mark (0, 0, 0) as the minimum
        ax.scatter([0], [0], [0], color='red', s=100, label='Minimum')

        # Add labels
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$f(x,y)$')
        ax.set_title('3D Surface of $f(x,y) = x^2 + xy + y^2$ with Gradient Vectors')

        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        return fig

    # 4. Saddle Point and Learning Rate Effects
    def plot_saddle_point():
        # Define a function with a saddle point
        def saddle_function(x, y):
            """Function with a saddle point at (0, 0)"""
            return x**2 - y**2

        # Create a grid for the saddle function - use fewer points for smaller file
        x = np.linspace(-2, 2, 80)
        y = np.linspace(-2, 2, 80)
        X, Y = np.meshgrid(x, y)
        Z = saddle_function(X, Y)

        # Create a 3D plot of the saddle function
        fig = plt.figure(figsize=(5, 4.5))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, 
                                linewidth=0, antialiased=True)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$f(x,y)$')
        ax.set_title('Saddle Point: $f(x,y) = x^2 - y^2$')

        # Mark the saddle point
        ax.scatter([0], [0], [0], color='red', s=100, label='Saddle Point')
        ax.legend()
        return fig

    def plot_learning_rate_effects():
        def gradient_descent_with_rate(start_point, learning_rate, iterations=50):
            """Compute gradient descent path with specified learning rate"""
            path = [np.array(start_point, dtype=float)]
            point = path[0].copy()
            
            for i in range(iterations):
                grad = grad_f(point[0], point[1])
                point = point - learning_rate * grad
                path.append(point.copy())
                    
            return np.array(path)

        # Plot contour lines
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        contour = plt.contour(X, Y, Z, 20, cmap=cm.viridis)
        plt.clabel(contour, inline=True, fontsize=8)

        # Compute and plot gradient descent paths with different learning rates
        start_point = (2, 2)
        learning_rates = [0.02, 0.1, 0.2, 0.5]
        colors = ['blue', 'green', 'orange', 'red']

        for i, lr in enumerate(learning_rates):
            path = gradient_descent_with_rate(start_point, lr)
            plt.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], 
                    label=f'η = {lr}', markersize=4, alpha=0.7)

        # Mark the minimum and starting point
        plt.plot(0, 0, 'r*', markersize=10, label='Minimum')
        plt.plot(start_point[0], start_point[1], 'ko', markersize=8, label='Start')

        # Add labels and title
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Effect of Learning Rate (η) on Gradient Descent')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    # Generate all visualizations
    if safe_plot(plot_gradient_field, "gradient_field.png"):
        successful += 1
    
    if safe_plot(plot_gradient_descent_paths, "gradient_descent_paths.png"):
        successful += 1
    
    try:
        fig = plot_3d_surface()
        fig.savefig(os.path.join(SAVE_DIR, "3d_surface_gradient.png"), 
                   dpi=120, bbox_inches='tight', pad_inches=0.2,
                   transparent=False)
        plt.close(fig)
        print(f"Successfully generated: 3d_surface_gradient.png")
        successful += 1
    except Exception as e:
        print(f"Error generating 3d_surface_gradient.png: {e}")
    
    try:
        fig = plot_saddle_point()
        fig.savefig(os.path.join(SAVE_DIR, "saddle_point.png"), 
                   dpi=120, bbox_inches='tight', pad_inches=0.2,
                   transparent=False)
        plt.close(fig)
        print(f"Successfully generated: saddle_point.png")
        successful += 0.5
    except Exception as e:
        print(f"Error generating saddle_point.png: {e}")
    
    if safe_plot(plot_learning_rate_effects, "learning_rate_effects.png"):
        successful += 0.5

    print(f"\nGenerated {successful}/{total} visualizations.")
    return successful == total

if __name__ == "__main__":
    print("Generating gradient visualizations...")
    success = generate_all_visualizations()
    
    if success:
        print("\nAll visualizations generated successfully!")
        print(f"Images saved to: {os.path.abspath(SAVE_DIR)}")
    else:
        print("\nSome visualizations could not be generated.")
        print("Please check the error messages above.")

