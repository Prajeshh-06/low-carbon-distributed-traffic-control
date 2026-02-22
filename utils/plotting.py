"""
Plotting Utilities
==================

Visualisation functions for traffic simulation results.
All functions accept the metrics dictionaries returned by
:func:`experiments.simulation.run_simulation`.
"""

import numpy as np
import matplotlib.pyplot as plt


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
