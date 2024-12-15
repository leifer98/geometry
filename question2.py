# Import required libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def generate_chessboard(chessboard_size=(8, 8), square_size=50):
    """
    Generate a grayscale chessboard image.
    
    Parameters:
        chessboard_size (tuple): Number of squares in (rows, columns).
        square_size (int): Pixel size of each square.
        
    Returns:
        ndarray: Normalized chessboard image (values between 0 and 1).
    """
    chessboard = np.zeros((chessboard_size[0] * square_size, chessboard_size[1] * square_size), dtype=np.uint8)
    for i in range(chessboard_size[0]):
        for j in range(chessboard_size[1]):
            if (i + j) % 2 == 0:
                chessboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 255
    return chessboard / 255.0  # Normalize to [0, 1]

def compute_laplacian(grid, laplacian_stencil, boundary='symm'):
    """
    Compute the discrete Laplacian of a 2D grid using convolution.
    
    Parameters:
        grid (ndarray): Input 2D grid.
        laplacian_stencil (ndarray): Discrete Laplacian stencil.
        boundary (str): Boundary condition ('fill', 'symm', 'wrap').
        
    Returns:
        ndarray: Laplacian of the grid.
    """
    return convolve2d(grid, laplacian_stencil, mode='same', boundary=boundary)

def heat_equation_solver(grid, laplacian_stencil, kappa, delta_t, time_steps, boundary='symm'):
    """
    Solve the heat equation using an explicit time-stepping scheme.
    
    Parameters:
        grid (ndarray): Initial grid (image).
        laplacian_stencil (ndarray): Discrete Laplacian stencil.
        kappa (float): Diffusion constant.
        delta_t (float): Time step size.
        time_steps (int): Number of time steps.
        boundary (str): Boundary condition ('fill', 'symm', 'wrap').
        
    Returns:
        list: List of grids at each time step.
    """
    u = grid.copy()
    delta_x = 1  # Assume unit spacing
    stability_condition = kappa * delta_t / (delta_x ** 2)
    if stability_condition > 0.25:
        raise ValueError(f"Stability condition violated: κ·Δt/(Δx)^2 = {stability_condition:.2f} > 0.25")
    
    grids_over_time = [u.copy()]
    for _ in range(time_steps):
        laplacian = compute_laplacian(u, laplacian_stencil, boundary=boundary)
        u += kappa * delta_t * laplacian
        grids_over_time.append(u.copy())
    return grids_over_time

def visualize_evolution(evolved_images, time_points):
    """
    Visualize the evolution of the image at selected time points.
    """
    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 5))
    for idx, t in enumerate(time_points):
        axes[idx].imshow(evolved_images[t], cmap='gray')
        axes[idx].set_title(f"t = {t}")
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

def compare_boundary_conditions(grid, laplacian_stencil, kappa, delta_t, time_steps):
    """
    Compare the effect of different boundary conditions on the heat equation solver.
    """
    evolved_dirichlet = heat_equation_solver(grid, laplacian_stencil, kappa, delta_t, time_steps, boundary='fill')
    evolved_neumann = heat_equation_solver(grid, laplacian_stencil, kappa, delta_t, time_steps, boundary='symm')
    evolved_periodic = heat_equation_solver(grid, laplacian_stencil, kappa, delta_t, time_steps, boundary='wrap')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(evolved_dirichlet[-1], cmap='gray')
    axes[0].set_title("Dirichlet Boundary (t = 50)")
    axes[0].axis("off")
    
    axes[1].imshow(evolved_neumann[-1], cmap='gray')
    axes[1].set_title("Neumann Boundary (t = 50)")
    axes[1].axis("off")
    
    axes[2].imshow(evolved_periodic[-1], cmap='gray')
    axes[2].set_title("Periodic Boundary (t = 50)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate the chessboard image
    gray_image = generate_chessboard()
    plt.figure(figsize=(6, 6))
    plt.title("Chessboard Grayscale Image")
    plt.imshow(gray_image, cmap='gray')
    plt.axis("off")
    plt.show()

    # Define the Laplacian stencil
    laplacian_stencil = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])

    # Heat equation parameters
    kappa = 0.5
    delta_t = 0.1
    time_steps = 50
    
    # Solve the heat equation
    evolved_images = heat_equation_solver(gray_image, laplacian_stencil, kappa, delta_t, time_steps)
    
    # Visualize the evolution
    time_points = [0, 10, 20, 30, 40, 50]
    visualize_evolution(evolved_images, time_points)
    
    # Compare boundary conditions
    compare_boundary_conditions(gray_image, laplacian_stencil, kappa, delta_t, time_steps)

if __name__ == "__main__":
    main()
