import os
import pandas as pd

# Set file paths
simid = "SCsurf_T"
restart_file = f"SkyrmionLattice/restart.{simid}.out"
exchange_file = "SkyrmionLattice/jij"
dm_file = "SkyrmionLattice/dmdata"
anisotropy_file = "SkyrmionLattice/anisotropy"  # optional

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# --- PARSE RESTART FILE ---
spins = []
with open(restart_file, 'r') as f:
    for line in f:
        tokens = line.split()
        if len(tokens) == 7:
            _, atom_type, site, mag, mx, my, mz = tokens
            spins.append({
                "site": int(site),
                "atom": int(atom_type),
                "mx": float(mx),
                "my": float(my),
                "mz": float(mz)
            })

spins_df = pd.DataFrame(spins).set_index("site")
print("\nSpins (first 5 rows):")
print(spins_df.head())

# SAVE SPINS TO CSV
spins_df.to_csv("data/parsed_restart.csv", index_label="site")
print(" Parsed spins saved to data/parsed_restart.csv")

# --- PARSE EXCHANGE FILE ---
exchange_df = pd.read_csv(exchange_file, delim_whitespace=True, names=["i", "j", "dx", "dy", "dz", "Jij"])
print("\nExchange (first 5 rows):")
print(exchange_df.head())

# --- PARSE DM FILE ---
dm_df = pd.read_csv(dm_file, delim_whitespace=True, names=["i", "j", "dx", "dy", "dz", "Dx", "Dy", "Dz"])
print("\nDM (first 5 rows):")
print(dm_df.head())

# --- OPTIONAL: PARSE ANISOTROPY FILE ---
if os.path.exists(anisotropy_file):
    aniso_df = pd.read_csv(anisotropy_file, delim_whitespace=True, header=None)
    print("\nAnisotropy file detected and loaded.")
else:
    print("\nNo Anisotropy file loaded.")

# ========== EXPORT FOR BENCHMARKING ==========
def load_system(base_dir):
    spins = pd.read_csv("data/parsed_restart.csv", index_col="site")

    exchange = pd.read_csv(
        os.path.join("SkyrmionLattice", "jij"),
        sep=r"\s+",
        names=["i", "j", "dx", "dy", "dz", "Jij"]
    )

    dm = pd.read_csv(
        os.path.join("SkyrmionLattice", "dmdata"),
        sep=r"\s+",
        names=["i", "j", "dx", "dy", "dz", "Dx", "Dy", "Dz"]
    )

    anisotropy_path = os.path.join("SkyrmionLattice", "anisotropy")
    anisotropy = None
    if os.path.exists(anisotropy_path):
        anisotropy = pd.read_csv(anisotropy_path, sep=r"\s+", header=None)

    return {
        "spins": spins,
        "exchange": exchange,
        "dm": dm,
        "anisotropy": anisotropy
    }

