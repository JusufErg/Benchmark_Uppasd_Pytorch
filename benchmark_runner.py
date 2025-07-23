import os
import pandas as pd
from parser import load_system
from optimizer import run_optimizer
from compare_spins import compare_spins

def run_benchmarks(base_dir, lr, steps, simid, restart_csv, optimizers):
    summary = []

    for opt in optimizers:
        opt_name = opt["name"]
        system = load_system(base_dir)

        optimized_spins = run_optimizer(
            system,
            lr=lr,
            steps=steps,
            optimizer_name=opt_name,
            simid=simid
        )

        opt_filename = f"data/optimized_spins_{simid}_{opt_name}.csv"
        optimized_df = pd.DataFrame(
            optimized_spins,  # assumed to be NumPy array or similar
            columns=["mx", "my", "mz"]
        )
        optimized_df.index = system["spins"].index
        optimized_df["atom"] = system["spins"]["atom"]
        optimized_df.to_csv(opt_filename)

        mean_diff, max_diff = compare_spins(
            restart_file=restart_csv,
            optimized_file=opt_filename
        )

        energy_log = pd.read_csv(f"data/energy_log_{simid}_{opt_name}.csv")
        final_energy = energy_log["total"].iloc[-1]

        summary.append({
            "optimizer": opt_name,
            "final_energy": final_energy,
            "mean_site_diff": mean_diff,
            "max_site_diff": max_diff,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"data/benchmark_summary_{simid}.csv", index=False)
    return summary_df

