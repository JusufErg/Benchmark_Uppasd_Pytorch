import pandas as pd
import numpy as np

def compare_spins(restart_file, optimized_file):
    """
    Compare spins between restart file and optimized result.
    Returns mean and max per-site difference (L2 norm).
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

    # Compute per-site differences (vector norms)
    diff_vectors = optimized_spins - restart_spins
    per_site_diff = np.linalg.norm(diff_vectors, axis=1)

    mean_diff = np.mean(per_site_diff)
    max_diff = np.max(per_site_diff)

    print(f" Mean per-site difference: {mean_diff:.6e}")
    print(f" Max per-site difference: {max_diff:.6e}")

    return mean_diff, max_diff

# Example usage (standalone)
if __name__ == "__main__":
    restart_file = "data/parsed_restart.csv"
    optimized_file = "data/optimized_spins_default_adam.csv"
    compare_spins(restart_file, optimized_file)

