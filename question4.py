# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Function to compute the heat kernel K(x, y, t)
def heat_kernel(x, y, source, t, alpha=1.0):
    """
    Compute the heat kernel K(x, y, t) in 2D space.

    Parameters:
        x (ndarray): X-coordinates of the grid points.
        y (ndarray): Y-coordinates of the grid points.
        source (tuple): Coordinates (y1, y2) of the heat source.
        t (float): Time.
        alpha (float): Thermal diffusivity (default: 1.0).
    
    Returns:
        ndarray: Heat kernel values at the grid points.
    """
    # Calculate squared distance from the heat source to grid points
    distance_squared = (x - source[0])**2 + (y - source[1])**2
    
    # Compute the heat kernel
    K = (1 / (4 * np.pi * alpha * t)) * np.exp(-distance_squared / (4 * alpha * t))
    return K

# Create a 2D grid of points
grid_size = 100  # Number of points along each axis
x = np.linspace(-5, 5, grid_size)  # X-coordinates
y = np.linspace(-5, 5, grid_size)  # Y-coordinates
X, Y = np.meshgrid(x, y)  # Create a mesh grid

# Parameters for the heat kernel
alpha = 1.0  # Thermal diffusivity (diffusion constant)
source_point = (0, 0)  # Heat source at origin (y = 0)
time_steps = [0.1, 1, 5, 10, 20]  # Time steps for simulation

# Cell 4: Heatmaps with constrained layout
fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 4), constrained_layout=True)
fig.suptitle("Heat Kernel Evolution Over Time", fontsize=16)

for i, t in enumerate(time_steps):
    K = heat_kernel(X, Y, source_point, t, alpha)
    ax = axes[i]
    im = ax.imshow(K, extent=(-5, 5, -5, 5), origin="lower", cmap="plasma", 
                   norm=Normalize(vmin=0, vmax=K.max()))
    ax.set_title(f"t = {t:.1f}")
    ax.axis("off")

fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.7)
plt.show()