import torch
import pandas as pd
from hamiltonian import full_spin_hamiltonian

def optimize_spins(spins_init, J_pairs, DMI_pairs=None, anisotropy_data=None, 
                   lr=0.01, steps=500, simid="default", optimizer_name="adam"):
    """
    Gradient-based spin optimization using PyTorch optimizers (Adam, SGD, LBFGS, RMSprop, etc.).
    Logs total and component energies at each step.
    """
    spins = spins_init.clone().detach().requires_grad_(True)

    # Choose optimizer
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam([spins], lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD([spins], lr=lr)
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS([spins], lr=lr, max_iter=20)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop([spins], lr=lr)
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad([spins], lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW([spins], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    history = {
        "step": [],
        "total": [],
        "heisenberg": [],
        "dmi": [],
        "anisotropy": []
    }

    for step in range(steps):
        def closure():
            optimizer.zero_grad()
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
            loss = terms["total"]
            loss.backward()
            if spins.grad is not None:
                spins.grad = spins.grad.contiguous()
            return loss

        if opt_name == "lbfgs":
            optimizer.step(closure)
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
        else:
            optimizer.zero_grad()
            spins_normed = spins / spins.norm(dim=1, keepdim=True)
            terms = full_spin_hamiltonian(spins_normed, J_pairs, DMI_pairs, anisotropy_data)
            terms["total"].backward()
            optimizer.step()
            with torch.no_grad():
                spins /= spins.norm(dim=1, keepdim=True)


        # Logging
        history["step"].append(step)
        history["total"].append(terms["total"].item())
        history["heisenberg"].append(terms["heisenberg"].item())
        history["dmi"].append(terms["dmi"].item())
        history["anisotropy"].append(terms["anisotropy"].item())


    # Save energy log
    df = pd.DataFrame(history)
    df.to_csv(f"data/energy_log_{simid}_{optimizer_name}.csv", index=False)
    print(f" Energy log saved to data/energy_log_{simid}_{optimizer_name}.csv")

    return spins.detach() / spins.detach().norm(dim=1, keepdim=True)

def run_optimizer(system, lr=0.1, steps=500, optimizer_name="adam", simid="default"):
    """
    Prepares system data and runs optimization.
    """
    spins_df = system["spins"]  # index=site, columns=['atom', 'mx', 'my', 'mz']
    exchange_df = system["exchange"]
    dm_df = system["dm"]
    aniso_df = system["anisotropy"]

    # Prepare initial spins (N, 3)
    spins_init = torch.tensor(spins_df[["mx", "my", "mz"]].values, dtype=torch.float32)

    # Prepare J_pairs: [i, j, Jij]
    J_pairs = torch.tensor(exchange_df[["i", "j", "Jij"]].values, dtype=torch.float32)

    # Prepare DMI_pairs if available: [i, j, Dx, Dy, Dz]
    DMI_pairs = None
    if dm_df is not None:
        DMI_pairs = torch.tensor(dm_df[["i", "j", "Dx", "Dy", "Dz"]].values, dtype=torch.float32)

    # Prepare anisotropy_data if available: [site, K1, ex, ey, ez]
    anisotropy_data = None
    if aniso_df is not None:
        anisotropy_data = torch.tensor(aniso_df[["site", "K1", "ex", "ey", "ez"]].values, dtype=torch.float32)

    # Run optimizer
    optimized_spins = optimize_spins(
        spins_init, J_pairs, DMI_pairs, anisotropy_data,
        lr=lr, steps=steps, simid=simid, optimizer_name=optimizer_name
    )

    return optimized_spins


