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
from utils.plotting import (
    compare_baseline_vs_carbon_aware,
    plot_emission,
    plot_queue_evolution,
)
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
    print("[1/3]  Running baseline simulation (γ = 0) …")
    baseline, carbon_aware = run_experiment(
        num_agents=NUM_AGENTS, steps=STEPS, seed=SEED
    )
    print("[2/3]  Running carbon-aware simulation (γ > 0) …")
    print("       Done.\n")

    # ---- Summary metrics ----
    total_emission_base = sum(baseline["emissions"])
    total_emission_ca = sum(carbon_aware["emissions"])
    avg_queue_base = float(np.mean(baseline["avg_queues"]))
    avg_queue_ca = float(np.mean(carbon_aware["avg_queues"]))
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
    base_recovery = baseline["avg_queues"][post_shock_window]
    ca_recovery = carbon_aware["avg_queues"][post_shock_window]
    print("  Shock Recovery (avg queue, steps 20–29):")
    print(f"    Baseline     :  {np.mean(base_recovery):.2f}")
    print(f"    Carbon-Aware :  {np.mean(ca_recovery):.2f}")
    print()

    # ---- Plots ----
    compare_baseline_vs_carbon_aware(baseline, carbon_aware)
    plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
    print("  Plot saved → comparison_plot.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close("all")


if __name__ == "__main__":
    main()
