"""
Plotting Utilities
==================

Visualisation functions for traffic simulation results.
All functions accept the metrics dictionaries returned by
:func:`experiments.simulation.run_simulation`.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiment import run_experiment
from experiments.simulation import run_simulation


def plot_emission(results, title="Total Emissions Over Time", ax=None):
    """
    Plot per-timestep total emissions.

    Parameters
    ----------
    results : dict
        Simulation output containing ``"emissions"`` key.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; creates a new figure if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results["emissions"], linewidth=1.8, color="#e74c3c")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Emission (carbon units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_queue_evolution(results, title="Average Queue Over Time", ax=None):
    """
    Plot per-timestep average queue length across all agents.

    Parameters
    ----------
    results : dict
        Simulation output containing ``"avg_queues"`` key.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; creates a new figure if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results["avg_queues"], linewidth=1.8, color="#3498db")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Queue Length")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def compare_baseline_vs_carbon_aware(baseline, carbon_aware):
    """
    Side-by-side comparison of baseline and carbon-aware simulations.

    Produces a 2×1 figure:
        Top    — emissions comparison
        Bottom — queue comparison

    Annotates total emission values and carbon savings percentage.

    Parameters
    ----------
    baseline : dict
        Results from the baseline (γ = 0) simulation.
    carbon_aware : dict
        Results from the carbon-aware (γ > 0) simulation.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- Emission comparison ----
    ax = axes[0]
    ax.plot(baseline["emissions"], label="Baseline (γ=0)", linewidth=1.8,
            color="#e74c3c", linestyle="--")
    ax.plot(carbon_aware["emissions"], label="Carbon-Aware", linewidth=1.8,
            color="#27ae60")
    ax.set_ylabel("Total Emission")
    ax.set_title("Emissions: Baseline vs Carbon-Aware")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate savings
    total_base = sum(baseline["emissions"])
    total_ca = sum(carbon_aware["emissions"])
    savings_pct = (1 - total_ca / total_base) * 100 if total_base > 0 else 0
    ax.annotate(
        f"Carbon savings: {savings_pct:.1f}%",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#d5f5e3", ec="#27ae60"),
    )

    # ---- Queue comparison ----
    ax = axes[1]
    ax.plot(baseline["avg_queues"], label="Baseline (γ=0)", linewidth=1.8,
            color="#e74c3c", linestyle="--")
    ax.plot(carbon_aware["avg_queues"], label="Carbon-Aware", linewidth=1.8,
            color="#27ae60")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Average Queue Length")
    ax.set_title("Queue: Baseline vs Carbon-Aware")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def generate_all_comparative_plots(results_dict):
    """
    Generate 12 comparative graphs and save them as PDFs in the /figures directory.
    """
    os.makedirs("figures", exist_ok=True)
    
    labels = {
        "baseline": "Baseline",
        "carbon": "Carbon-Aware",
        "high_gamma": "High Sustain (gamma=1.5)",
        "short_horizon": "Short Horizon (H=3)",
        "long_horizon": "Long Horizon (H=10)",
        "lstm_carbon": "LSTM + Carbon-Aware",
    }
    
    colors = {
        "baseline": "#e74c3c",
        "carbon": "#27ae60",
        "high_gamma": "#f39c12",
        "short_horizon": "#8e44ad",
        "long_horizon": "#2980b9",
        "lstm_carbon": "#e91e63",
    }

    # Helper function for consistently styling and saving plots
    def save_fig(fig, ax, filename, title, xlabel, ylabel):
        ax.set_title(title, pad=15, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join("figures", filename), format="pdf")
        plt.close(fig)

    # 1. Emission vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        ax.plot(res["total_emission"], label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig1_emission_time.pdf", "Emission vs Time", "Timestep", "Total Emission (carbon units)")
    
    # 2. Queue vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        ax.plot(res["avg_queue"], label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig2_queue_time.pdf", "Queue vs Time", "Timestep", "Average Queue Length")

    # 3. Green Time vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        avg_green = [np.mean(step) for step in res["green_time_trajectory"]]
        ax.plot(avg_green, label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig3_green_time.pdf", "Green Time vs Time", "Timestep", "Average Green Time Fraction")

    # 4. Gamma Convergence vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        if "gamma_trajectory" in res:
            avg_gamma = [np.mean(step) for step in res["gamma_trajectory"]]
            ax.plot(avg_gamma, label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig4_gamma_time.pdf", "Gamma Convergence vs Time", "Timestep", "Average Gamma Value")

    # 5. Cumulative Emission vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        cum_emission = np.cumsum(res["total_emission"])
        ax.plot(cum_emission, label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig5_cumulative_emission.pdf", "Cumulative Emission vs Time", "Timestep", "Cumulative Emission")

    # 6. Total Emission Bar Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    keys = list(results_dict.keys())
    totals = [sum(results_dict[k]["total_emission"]) for k in keys]
    ax.bar(keys, totals, color=[colors.get(k, "gray") for k in keys])
    ax.set_title("Total Emission Bar Comparison", pad=15, fontweight="bold")
    ax.set_ylabel("Total Sum of Emissions")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "fig6_total_emission_bar.pdf"), format="pdf")
    plt.close(fig)

    # 7. Average Queue Bar Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_q = [np.mean(results_dict[k]["avg_queue"]) for k in keys]
    ax.bar(keys, avg_q, color=[colors.get(k, "gray") for k in keys])
    ax.set_title("Average Queue Bar Comparison", pad=15, fontweight="bold")
    ax.set_ylabel("Mean Average Queue")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "fig7_avg_queue_bar.pdf"), format="pdf")
    plt.close(fig)

    # 8. Shock Response Zoom (Emission between t=15 to t=30)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in results_dict.items():
        # Make sure enough timesteps exist
        if len(res["total_emission"]) >= 31:
            emissions_zoom = res["total_emission"][15:31]
            ax.plot(range(15, 31), emissions_zoom, label=labels.get(key, key), color=colors.get(key), linewidth=2, marker="o")
    save_fig(fig, ax, "fig8_shock_zoom.pdf", "Shock Response Zoom (Emission between t=15 to t=30)", "Timestep", "Total Emission (carbon units)")

    # 9. Horizon vs Total Emission
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = [3, 5, 10]
    ems = [
        sum(results_dict["short_horizon"]["total_emission"]),
        sum(results_dict["carbon"]["total_emission"]),
        sum(results_dict["long_horizon"]["total_emission"])
    ]
    ax.plot(horizons, ems, marker="s", linestyle="-", linewidth=2, color="#2c3e50", label="Emission vs Horizon")
    save_fig(fig, ax, "fig9_horizon_emission.pdf", "Horizon vs Total Emission", "Prediction Horizon (H)", "Total Emission")

    # 10. Gamma vs Total Emission
    fig, ax = plt.subplots(figsize=(8, 5))
    gammas = [0.0, 0.9, 1.5]
    ems_gamma = [
        sum(results_dict["baseline"]["total_emission"]),
        sum(results_dict["carbon"]["total_emission"]),
        sum(results_dict["high_gamma"]["total_emission"])
    ]
    ax.plot(gammas, ems_gamma, marker="^", linestyle="-", linewidth=2, color="#16a085", label="Emission vs Gamma")
    save_fig(fig, ax, "fig10_gamma_emission.pdf", "Gamma vs Total Emission", "Initial Gamma Coefficient", "Total Emission")

    # 11. Emission vs Shock Factor (simulate factors 1-3)
    shock_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    results_shock = {k: [] for k in results_dict.keys()}
    
    # Minimal runs to gather total emissions for different shocks
    for sf in shock_factors:
        res_sf = run_experiment(steps=50, shock_factor=sf, shock_step=20)
        for k in results_dict.keys():
            results_shock[k].append(sum(res_sf[k]["total_emission"]))
            
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in results_dict.keys():
        ax.plot(shock_factors, results_shock[key], marker="o", label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig11_shock_factor.pdf", "Emission vs Shock Factor", "Shock Factor", "Total Emission")

    # 12. Computational Complexity vs Horizon (measure runtime)
    test_horizons = [2, 4, 6, 8, 10]
    runtimes_horizon = {k: [] for k in results_dict.keys()}
    
    # We will test run_simulation direct approach for 15 steps
    for h in test_horizons:
        for k in runtimes_horizon.keys():
            gamma_val = 1.5 if k == "high_gamma" else (0.0 if k == "baseline" else 0.9)
            fed_sync = False if k == "baseline" else True
            
            start_t = time.time()
            run_simulation(steps=15, gamma=gamma_val, federated_sync=fed_sync, horizon=h)
            dt = time.time() - start_t
            runtimes_horizon[k].append(dt)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in runtimes_horizon.keys():
        ax.plot(test_horizons, runtimes_horizon[key], marker="d", label=labels.get(key, key), color=colors.get(key), linewidth=2)
    save_fig(fig, ax, "fig12_complexity.pdf", "Computational Complexity", "Prediction Horizon (H)", "Simulation Runtime (seconds)")
