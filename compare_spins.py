import numpy as np
import pandas as pd

def compare_spins(restart_file, optimized_file):
    """
    Compare spins between restart file and optimized result.
    Returns mean and max per-site angular deviation (in degrees).
    Assumes all spins are normalized.
    """
    # Load files
    restart_df = pd.read_csv(restart_file, index_col="site")
    optimized_df = pd.read_csv(optimized_file, index_col="site")

    # Align order
    restart_df = restart_df.sort_index()
    optimized_df = optimized_df.sort_index()

    # Extract mx, my, mz as numpy arrays
    restart_spins = restart_df[["mx", "my", "mz"]].values
    optimized_spins = optimized_df[["mx", "my", "mz"]].values

    # Normalize (in case of slight numerical drift)
    restart_spins = restart_spins / np.linalg.norm(restart_spins, axis=1, keepdims=True)
    optimized_spins = optimized_spins / np.linalg.norm(optimized_spins, axis=1, keepdims=True)

    # Compute cosine similarity and clip to avoid numerical errors
    dot_products = np.sum(restart_spins * optimized_spins, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Compute angles in degrees
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)

    mean_angle = np.mean(angles_deg)
    max_angle = np.max(angles_deg)

    print(f" Mean per-site angular deviation: {mean_angle:.4f}°")
    print(f" Max per-site angular deviation: {max_angle:.4f}°")

    return mean_angle, max_angle

# Example usage
if __name__ == "__main__":
    restart_file = "data/parsed_restart.csv"
    optimized_file = "data/optimized_spins_default_adam.csv"
    compare_spins(restart_file, optimized_file)
