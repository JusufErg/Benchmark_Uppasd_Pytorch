import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
from benchmark_runner import run_benchmarks

def batch_run(base_dir, lr, steps, simid, restart_csv, optimizers, runs=10):
    os.makedirs("data", exist_ok=True)
    all_runs = []
    timing_log = []

    batch_start = time.time()
    for i in range(runs):
        run_id = f"{simid}_run{i}"
        print(f"=== Starting run {i+1}/{runs} ===")

        for opt in optimizers:
            opt_name = opt["name"]
            opt_params = opt.get("params", {})
            print(f"Running {opt_name}...")

            start = time.time()
            df = run_benchmarks(base_dir, lr, steps, run_id, restart_csv, [opt])
            elapsed = round(time.time() - start, 4)

            df["run"] = i
            all_runs.append(df)
            print(f"Optimizer {opt_name} finished in {elapsed:.2f} seconds")

            timing_log.append({
                "optimizer": opt_name,
                "run": i,
                "time": elapsed
            })

    batch_end = time.time()
    print(f"Total batch time: {batch_end - batch_start:.2f} seconds")

    # Save batch result
    combined_df = pd.concat(all_runs, ignore_index=True)
    combined_df.to_csv(f"data/batch_summary_{simid}.csv", index=False)

    # Save timing CSV + plot
    timing_df = pd.DataFrame(timing_log)
    timing_df.to_csv("data/timing_log.csv", index=False)
    plot_timing(timing_df, simid)

    return combined_df

def plot_timing(df, simid):
    plt.figure(figsize=(8, 5))
    for opt in df["optimizer"].unique():
        sub = df[df["optimizer"] == opt]
        plt.plot(sub["run"], sub["time"], label=opt, marker='o')

    plt.title("Runtime vs Run ID")
    plt.xlabel("Run (Seed ID)")
    plt.ylabel("Time (s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Optimizer")
    plt.tight_layout()
    plt.savefig(f"data/{simid}_timing_plot.png")
    plt.close()
    print(f" Saved timing plot to: data/{simid}_timing_plot.png")

def plot_boxplots(df, simid):
    metrics = ["final_energy", "Mean per-site angular deviation (°)", "Max per-site angular deviation (°)"]
    for metric in metrics:
        plt.figure()
        df.boxplot(column=metric, by="optimizer")
        plt.title(f"{metric} by optimizer")
        plt.suptitle("")
        plt.ylabel(metric)
        plt.xlabel("Optimizer")
        plt.tight_layout()
        plt.savefig(f"data/{simid}_{metric}_boxplot.png")
        plt.close()

if __name__ == "__main__":
    BASE_DIR = "SkyrmionLattice"
    LR = 0.01
    STEPS = 5000
    SIMID = "SkyrmionTest"
    RESTART_CSV = "data/parsed_restart.csv"
    RUNS = 3

    OPTIMIZERS = [
        {"name": "adam", "params": {}},
        {"name": "sgd", "params": {}},
        {"name": "lbfgs", "params": {}},
        {"name": "rmsprop", "params": {}},
        {"name": "adagrad", "params": {}},
        {"name": "adamw", "params": {}},
    ]

    combined_df = batch_run(BASE_DIR, LR, STEPS, SIMID, RESTART_CSV, OPTIMIZERS, runs=RUNS)
    plot_boxplots(combined_df, SIMID)
