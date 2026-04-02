"""
Plotting Utilities
==================

Generates 12 diverse, print-quality comparative figures for the IEEE journal paper.
All figures are exported as PDF to the /figures directory.

Colour palette: muted, low-saturation, print-friendly.
Font sizes: large enough for comfortable reading when printed at two-column width.
"""

import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from experiments.run_experiment import run_experiment
from experiments.simulation import run_simulation

# ─────────────────────────────────────────────
#  Global style settings  (bumped from previous version)
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         15,        # base (was 13)
    "axes.titlesize":    17,        # unused — titles removed from graphs
    "axes.labelsize":    15,        # axis labels (was 13)
    "xtick.labelsize":   14,        # tick labels (was 12)
    "ytick.labelsize":   14,
    "legend.fontsize":   13,        # legend (was 11)
    "legend.framealpha": 0.85,
    "lines.linewidth":   2.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# ─────────────────────────────────────────────
#  Muted colour palette
# ─────────────────────────────────────────────
COLORS = {
    "baseline":      "#5d6d7e",   # slate grey
    "carbon":        "#5d8a6e",   # muted sage green
    "high_gamma":    "#a07840",   # muted amber
    "short_horizon": "#6b5b8c",   # muted mauve
    "long_horizon":  "#3d6b8f",   # steel blue
    "lstm_carbon":   "#8c5b5b",   # dusty rose
}

LABELS = {
    "baseline":      r"Baseline MPC ($\gamma=0$)",
    "carbon":        r"Carbon-Aware MPC ($\gamma=0.9$)",
    "high_gamma":    r"High Sustainability ($\gamma=1.5$)",
    "short_horizon": r"Short Horizon ($H=3$)",
    "long_horizon":  r"Long Horizon ($H=10$)",
    "lstm_carbon":   r"LSTM + Carbon-Aware",
}

ORDERED_KEYS = ["baseline", "carbon", "high_gamma", "short_horizon",
                "long_horizon", "lstm_carbon"]

FIGURES_DIR = "figures"


def _save(fig, filename):
    """Tight-layout then save as PDF."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), format="pdf",
                bbox_inches="tight")
    plt.close(fig)


def generate_all_comparative_plots(results_dict):
    """
    Generate 12 diverse comparative graphs and save as PDFs in /figures.

    Graph types include:
        line plots, bar charts, a heat-map matrix, a scatter plot,
        a cumulative area chart, a horizon-sensitivity curve,
        a gamma-sensitivity curve, a shock-factor plot,
        and a runtime complexity plot.

    Parameters
    ----------
    results_dict : dict
        Mapping of configuration name -> simulation result dict, as returned
        by :func:`experiments.run_experiment.run_experiment`.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    keys = ORDERED_KEYS

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    # Line plot: Total emission over time (all 6 configs)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        ax.plot(results_dict[k]["total_emission"],
                color=COLORS[k], label=LABELS[k])
    ax.axvline(x=20, color="#999999", linestyle=":", linewidth=1.5,
               label="Demand shock ($t=20$)")
    ax.set_xlabel("Simulation Timestep $t$")
    ax.set_ylabel(r"Network Emission $\sum_i E_{i,t}$ (carbon units)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig1_emission_time.pdf")

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    # Line plot: Average queue length over time
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        ax.plot(results_dict[k]["avg_queue"],
                color=COLORS[k], label=LABELS[k])
    ax.axvline(x=20, color="#999999", linestyle=":", linewidth=1.5,
               label="Demand shock ($t=20$)")
    ax.set_xlabel("Simulation Timestep $t$")
    ax.set_ylabel(r"Mean Queue Length $\bar{q}_t$ (vehicles)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig2_queue_time.pdf")

    # ── Figure 3 ─────────────────────────────────────────────────────────────
    # Horizontal grouped bar chart: total emission + avg queue
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    short_labels = [LABELS[k].split("(")[0].strip() for k in keys]
    totals  = [sum(results_dict[k]["total_emission"]) for k in keys]
    avg_q   = [float(np.mean(results_dict[k]["avg_queue"])) for k in keys]
    bar_cols = [COLORS[k] for k in keys]
    y = np.arange(len(keys))

    axes[0].barh(y, totals, color=bar_cols, edgecolor="#333333", linewidth=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels(short_labels, fontsize=13)
    axes[0].set_xlabel("Total Cumulative Emission")
    axes[0].grid(True, alpha=0.25, axis="x", linestyle="--")
    base_tot = totals[0]
    for i, v in enumerate(totals):
        pct = (1 - v / base_tot) * 100 if i > 0 else 0
        axes[0].text(v + 5, i, f"−{pct:.1f}%" if i > 0 else "Ref",
                     va="center", fontsize=12, color="#444444")

    axes[1].barh(y, avg_q, color=bar_cols, edgecolor="#333333", linewidth=0.6)
    axes[1].set_yticks(y); axes[1].set_yticklabels(short_labels, fontsize=13)
    axes[1].set_xlabel("Mean Queue Length (vehicles)")
    axes[1].grid(True, alpha=0.25, axis="x", linestyle="--")

    _save(fig, "fig3_green_time.pdf")

    # ── Figure 4 ─────────────────────────────────────────────────────────────
    # Gamma convergence line plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        if "gamma_trajectory" in results_dict[k]:
            avg_g = [float(np.mean(step))
                     for step in results_dict[k]["gamma_trajectory"]]
            ax.plot(avg_g, color=COLORS[k], label=LABELS[k])
    ax.set_xlabel("Simulation Timestep $t$")
    ax.set_ylabel(r"Mean Carbon Penalty $\bar{\gamma}_t$")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig4_gamma_time.pdf")

    # ── Figure 5 ─────────────────────────────────────────────────────────────
    # Cumulative emission area chart
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        cum = np.cumsum(results_dict[k]["total_emission"])
        ax.plot(cum, color=COLORS[k], label=LABELS[k])
        ax.fill_between(range(len(cum)), cum, alpha=0.10, color=COLORS[k])
    ax.set_xlabel("Simulation Timestep $t$")
    ax.set_ylabel("Cumulative Emission (carbon units)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig5_cumulative_emission.pdf")

    # ── Figure 6 ─────────────────────────────────────────────────────────────
    # HEAT MAP: Per-config emission intensity across 10-step windows
    ca_result = results_dict["carbon"]
    T = len(ca_result["total_emission"])
    window = max(1, T // 10)
    configs_short = ["Baseline", "Carbon-Aware", "High Sust.",
                     "Short H", "Long H", "LSTM+CA"]
    heatmap_data = []
    for k in keys:
        row = []
        for w in range(10):
            t0, t1 = w * window, min((w + 1) * window, T)
            seg = results_dict[k]["total_emission"][t0:t1]
            row.append(float(np.mean(seg)) if seg else 0.0)
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Emission per Window", fontsize=14)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"W{i+1}" for i in range(10)], fontsize=13)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(configs_short, fontsize=13)
    ax.set_xlabel("10-Timestep Simulation Windows")
    ax.set_ylabel("Controller Configuration")
    for i in range(len(keys)):
        for j in range(10):
            ax.text(j, i, f"{heatmap_data[i, j]:.1f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if heatmap_data[i, j] > heatmap_data.max() * 0.6 else "#222222")
    _save(fig, "fig6_total_emission_bar.pdf")

    # ── Figure 7 ─────────────────────────────────────────────────────────────
    # SCATTER PLOT: Queue vs. Emission Pareto cloud
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in keys:
        qs = results_dict[k]["avg_queue"]
        em = results_dict[k]["total_emission"]
        ax.scatter(qs, em, color=COLORS[k], alpha=0.45, s=35,
                   edgecolors="none", label=LABELS[k])
    ax.set_xlabel(r"Average Queue Length $\bar{q}_t$ (vehicles)")
    ax.set_ylabel(r"Total Emission $\sum_i E_{i,t}$ (carbon units)")
    ax.legend(loc="upper left", ncol=1, markerscale=1.8)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig7_avg_queue_bar.pdf")

    # ── Figure 8 ─────────────────────────────────────────────────────────────
    # SHOCK ZOOM — emission between t=15..35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    t_start, t_end = 15, 36
    for k in keys:
        em = results_dict[k]["total_emission"]
        if len(em) >= t_end:
            ax.plot(range(t_start, t_end), em[t_start:t_end],
                    color=COLORS[k], marker="o", markersize=5,
                    label=LABELS[k])
    ax.axvline(x=20, color="#888888", linestyle=":", linewidth=1.5,
               label="Shock onset ($t=20$)")
    ax.set_xlabel("Simulation Timestep $t$")
    ax.set_ylabel(r"Network Emission $\sum_i E_{i,t}$")
    ax.legend(ncol=2, fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig8_shock_zoom.pdf")

    # ── Figure 9 ─────────────────────────────────────────────────────────────
    # HORIZON SENSITIVITY — total emission vs prediction horizon H
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons = [3, 5, 10]
    ems = [
        sum(results_dict["short_horizon"]["total_emission"]),
        sum(results_dict["carbon"]["total_emission"]),
        sum(results_dict["long_horizon"]["total_emission"]),
    ]
    ax.plot(horizons, ems, color=COLORS["carbon"], marker="s",
            markersize=10, linestyle="-", linewidth=2.2,
            label=r"Carbon-Aware MPC ($\gamma=0.9$)")
    for h, e in zip(horizons, ems):
        ax.annotate(f"{e:.1f}", xy=(h, e), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=13)
    ax.set_xlabel("Prediction Horizon $H$ (timesteps)")
    ax.set_ylabel("Total Cumulative Emission")
    ax.set_xticks(horizons)
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig9_horizon_emission.pdf")

    # ── Figure 10 ────────────────────────────────────────────────────────────
    # GAMMA vs TOTAL EMISSION — parametric curve
    fig, ax = plt.subplots(figsize=(8, 5))
    gammas = [0.0, 0.9, 1.5]
    ems_g = [
        sum(results_dict["baseline"]["total_emission"]),
        sum(results_dict["carbon"]["total_emission"]),
        sum(results_dict["high_gamma"]["total_emission"]),
    ]
    ax.plot(gammas, ems_g, color=COLORS["high_gamma"], marker="^",
            markersize=11, linestyle="-", linewidth=2.2)
    for g, e in zip(gammas, ems_g):
        ax.annotate(f"{e:.1f}", xy=(g, e), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=13)
    ax.set_xlabel(r"Carbon Penalty Coefficient $\gamma$")
    ax.set_ylabel("Total Cumulative Emission")
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig10_gamma_emission.pdf")

    # ── Figure 11 ────────────────────────────────────────────────────────────
    # EMISSION vs SHOCK FACTOR multi-series line plot
    shock_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    results_shock = {k: [] for k in keys}
    for sf in shock_factors:
        res_sf = run_experiment(steps=50, shock_factor=sf, shock_step=20)
        for k in keys:
            results_shock[k].append(sum(res_sf[k]["total_emission"]))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        ax.plot(shock_factors, results_shock[k],
                color=COLORS[k], marker="o", markersize=7,
                label=LABELS[k])
    ax.set_xlabel(r"Shock Multiplier $\kappa = \lambda_{\mathrm{shock}} / \lambda$")
    ax.set_ylabel("Total Cumulative Emission")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig11_shock_factor.pdf")

    # ── Figure 12 ────────────────────────────────────────────────────────────
    # COMPUTATIONAL COMPLEXITY — runtime vs horizon with error bands
    test_horizons = [2, 3, 5, 7, 10]
    N_REPEATS = 5
    runtime_matrix = {k: np.zeros((N_REPEATS, len(test_horizons)))
                      for k in keys}

    gamma_map = {"baseline": 0.0, "carbon": 0.9, "high_gamma": 1.5,
                 "short_horizon": 0.9, "long_horizon": 0.9, "lstm_carbon": 0.9}
    fed_map   = {k: False if k == "baseline" else True for k in keys}

    for r in range(N_REPEATS):
        for hi, h in enumerate(test_horizons):
            for k in keys:
                t0 = time.perf_counter()
                run_simulation(steps=15, gamma=gamma_map[k],
                               federated_sync=fed_map[k], horizon=h, seed=r)
                runtime_matrix[k][r, hi] = time.perf_counter() - t0

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in keys:
        mean_rt = runtime_matrix[k].mean(axis=0)
        std_rt  = runtime_matrix[k].std(axis=0)
        ax.plot(test_horizons, mean_rt, color=COLORS[k],
                marker="d", markersize=7, label=LABELS[k])
        ax.fill_between(test_horizons,
                        mean_rt - std_rt, mean_rt + std_rt,
                        alpha=0.15, color=COLORS[k])
    ax.set_xlabel("Prediction Horizon $H$ (timesteps)")
    ax.set_ylabel("Wall-Clock Runtime (seconds)")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig12_complexity.pdf")

    print("Done! All 12 figures saved to ./figures/")


# ─────────────────────────────────────────────
#  Legacy helpers (retained for compatibility)
# ─────────────────────────────────────────────

def plot_emission(results, title="Total Emissions Over Time", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(results["emissions"], linewidth=2.0, color=COLORS["carbon"])
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Total Emission (carbon units)")
    ax.grid(True, alpha=0.25, linestyle="--")


def plot_queue_evolution(results, title="Average Queue Over Time", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(results["avg_queues"], linewidth=2.0, color=COLORS["baseline"])
    ax.set_xlabel("Timestep $t$")
    ax.set_ylabel("Average Queue Length (vehicles)")
    ax.grid(True, alpha=0.25, linestyle="--")
