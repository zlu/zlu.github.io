import matplotlib.pyplot as plt
import numpy as np

def draw_tree(x, y, angle, depth, length):
    if depth == 0:
        return
    x2 = x + length * np.cos(angle)
    y2 = y + length * np.sin(angle)
    plt.plot([x, x2], [y, y2], color='green')
    
    # Left branch
    draw_tree(x2, y2, angle + np.pi/6, depth - 1, length * 0.7)
    # Right branch
    draw_tree(x2, y2, angle - np.pi/6, depth - 1, length * 0.7)

plt.figure(figsize=(8, 8))
draw_tree(0, 0, np.pi/2, depth=7, length=100)
plt.axis('off')
plt.title("Fractal Tree")
plt.show()