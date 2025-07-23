import numpy as np
import pandas as pd
import os

def parse_restart(restart_file):
    data = np.loadtxt(restart_file)
    # Columns: iterens, iatom, site, |Mom|, M_x, M_y, M_z
    iatom = data[:, 1].astype(int)
    site = data[:, 2].astype(int)
    mx = data[:, 4]
    my = data[:, 5]
    mz = data[:, 6]
    spins = pd.DataFrame({
        "atom": iatom,
        "site": site,
        "mx": mx,
        "my": my,
        "mz": mz
    })
    return spins.set_index("site")

def parse_exchange(exchange_file):
    data = np.loadtxt(exchange_file)
    df = pd.DataFrame(data[:, :6], columns=["i", "j", "dx", "dy", "dz", "Jij"])
    df[["i", "j"]] = df[["i", "j"]].astype(int)
    return df

def parse_dm(dm_file):
    if not os.path.isfile(dm_file):
        print(f"[INFO] DM file '{dm_file}' not found. Skipping.")
        return None
    data = np.loadtxt(dm_file)
    df = pd.DataFrame(data, columns=["i", "j", "dx", "dy", "dz", "Dx", "Dy", "Dz"])
    df[["i", "j"]] = df[["i", "j"]].astype(int)
    return df

def parse_anisotropy(anisotropy_file):
    if not os.path.isfile(anisotropy_file):
        print(f"[INFO] Anisotropy file '{anisotropy_file}' not found. Skipping.")
        return None
    data = np.loadtxt(anisotropy_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    columns = ["site", "mode", "K1", "K2", "ex", "ey", "ez"]
    if data.shape[1] == 8:
        columns.append("ratio")
    df = pd.DataFrame(data, columns=columns)
    df["site"] = df["site"].astype(int)
    return df

def load_system(base_dir):
    restart_file = os.path.join(base_dir, "restart.SCsurf_T.out")
    exchange_file = os.path.join(base_dir, "jij")
    dm_file = os.path.join(base_dir, "dmdata")

    # Auto-detect anisotropy file if exists
    anisotropy_file = None
    for f in os.listdir(base_dir):
        if f.startswith("anisotropy"):
            anisotropy_file = os.path.join(base_dir, f)
            break

    spins = parse_restart(restart_file)
    exchange = parse_exchange(exchange_file)
    dm = parse_dm(dm_file)
    anisotropy = parse_anisotropy(anisotropy_file) if anisotropy_file else None

    system = {
        "spins": spins,
        "exchange": exchange,
        "dm": dm,
        "anisotropy": anisotropy
    }
    return system

if __name__ == "__main__":
    base_path = "SkyrmionLattice"
    system = load_system(base_path)

    print("\n Spins (first 5 rows):\n", system["spins"].head())
    print("\n Exchange (first 5 rows):\n", system["exchange"].head())
    if system["dm"] is not None:
        print("\n DM (first 5 rows):\n", system["dm"].head())
    else:
        print("\n No DM file loaded.")
    if system["anisotropy"] is not None:
        print("\n Anisotropy (first 5 rows):\n", system["anisotropy"].head())
    else:
        print("\n No Anisotropy file loaded.")

