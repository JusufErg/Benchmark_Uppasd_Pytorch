import torch

# Define global external magnetic field (Tesla units assumed)
#hfield = torch.tensor([0.0, 0.0, 0.0])  # Example: field along z-axis

def full_spin_hamiltonian(spins, J_pairs, DMI_pairs=None, anisotropy_data=None, B_ext=None):
    """
    Compute total energy and return breakdown of Heisenberg, DMI, Anisotropy, and External Field contributions.

    Args:
        spins: Tensor of shape (N, 3) for spin vectors.
        J_pairs: Tensor of shape (M, 3) where each row is (i, j, Jij).
        DMI_pairs: Tensor of shape (K, 5) where each row is (i, j, Dx, Dy, Dz). Default is None.
        anisotropy_data: Tensor of shape (L, 5) where each row is (i, Ki, nx, ny, nz). Default is None.
        B_ext: Tensor of shape (3,) representing a constant external magnetic field. Default is `hfield`.

    Returns:
        dict with keys: 'total', 'heisenberg', 'dmi', 'anisotropy', 'external'
    """
    e_heis = e_dmi = e_aniso = e_ext = torch.tensor(0.0, device=spins.device)

    # Heisenberg exchange
    i, j = J_pairs[:, 0].long(), J_pairs[:, 1].long()
    Jij = J_pairs[:, 2]
    e_heis = -torch.sum(Jij * torch.sum(spins[i] * spins[j], dim=1))

    # DMI
    if DMI_pairs is not None:
        i_dmi = DMI_pairs[:, 0].long()
        j_dmi = DMI_pairs[:, 1].long()
        Dij = DMI_pairs[:, 2:5]
        cross = torch.cross(spins[i_dmi], spins[j_dmi], dim=1)
        e_dmi = -torch.sum(torch.sum(Dij * cross, dim=1))

    # Anisotropy
    if anisotropy_data is not None:
        ai = anisotropy_data[:, 0].long()
        Ki = anisotropy_data[:, 1]
        ni = anisotropy_data[:, 2:5]
        proj = torch.sum(spins[ai] * ni, dim=1)
        e_aniso = -torch.sum(Ki * proj ** 2)

    # External field
    if B_ext is not None:
        e_ext = -torch.sum(spins @ B_ext)

    total = e_heis + e_dmi + e_aniso + e_ext

    return {
        "total": total,
        "heisenberg": e_heis,
        "dmi": e_dmi,
        "anisotropy": e_aniso,
        "external": e_ext
    }

