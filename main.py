"""
Low-Carbon Distributed Traffic Signal Control
==============================================

Entry point.  Runs baseline and carbon-aware multi-agent MPC simulations,
prints summary metrics, and displays comparison plots.

Usage
-----
    python main.py
"""

import numpy as np

from experiments.run_experiment import run_experiment
from utils.plotting import generate_all_comparative_plots
import matplotlib.pyplot as plt


def main():
    # ---- Configuration ----
    NUM_AGENTS = 4
    STEPS = 50
    SEED = 42

    print("=" * 62)
    print("  Low-Carbon Distributed Traffic Signal Control")
    print("  Multi-Agent MPC  ·  Federated Learning  ·  Carbon-Aware")
    print("=" * 62)
    print()

    # ---- Run experiments ----
    print("[1/3]  Running simulations (all configs) …")
    results = run_experiment(
        num_agents=NUM_AGENTS, steps=STEPS, seed=SEED
    )
    baseline = results["baseline"]
    carbon_aware = results["carbon"]
    print("       Done.\n")

    # ---- Summary metrics ----
    total_emission_base = sum(baseline["total_emission"])
    total_emission_ca = sum(carbon_aware["total_emission"])
    avg_queue_base = float(np.mean(baseline["avg_queue"]))
    avg_queue_ca = float(np.mean(carbon_aware["avg_queue"]))
    savings_pct = (
        (1 - total_emission_ca / total_emission_base) * 100
        if total_emission_base > 0
        else 0.0
    )

    print("[3/3]  Summary Metrics")
    print("-" * 62)
    print(f"  {'Metric':<30} {'Baseline':>12} {'Carbon-Aware':>14}")
    print("-" * 62)
    print(f"  {'Total Emission':<30} {total_emission_base:>12.2f} {total_emission_ca:>14.2f}")
    print(f"  {'Average Queue Length':<30} {avg_queue_base:>12.2f} {avg_queue_ca:>14.2f}")
    print(f"  {'Carbon Savings (%)':<30} {'—':>12} {savings_pct:>13.1f}%")
    print("-" * 62)
    print()

    # ---- Shock recovery ----
    shock_idx = 20
    post_shock_window = slice(shock_idx, min(shock_idx + 10, STEPS))
    base_recovery = baseline["avg_queue"][post_shock_window]
    ca_recovery = carbon_aware["avg_queue"][post_shock_window]
    print("  Shock Recovery (avg queue, steps 20–29):")
    print(f"    Baseline     :  {np.mean(base_recovery):.2f}")
    print(f"    Carbon-Aware :  {np.mean(ca_recovery):.2f}")
    print()

    # ---- Plots ----
    print("[4/4]  Generating plots to 'figures/' directory ...")
    generate_all_comparative_plots(results)


if __name__ == "__main__":
    main()
