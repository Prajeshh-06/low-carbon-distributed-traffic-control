"""
Smart City Low-Carbon Traffic Control Center
============================================
Streamlit dashboard for interactive multi-agent MPC simulation.

Run with:
    streamlit run dashboard.py
"""

import sys
import pathlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root is on the import path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from federated.agent import IntersectionAgent
from federated.fedavg import fedavg
from experiments.shock_events import apply_shock

# ── Shared simulation constants ───────────────────────────────────────────────
_STEPS         = 50
_SEED          = 42
_CAPACITY      = 15.0
_MEAN_DEMAND   = 7.0
_CARBON_BUDGET = 15.0
_SHOCK_STEP    = 20
_FEDAVG_INTV   = 5


def _run(gamma_val: float, num_agents: int, horizon: int, shock_factor: float) -> dict:
    """
    Run a single simulation for the given parameters.

    gamma_val = 0.0  →  baseline (no carbon awareness)
    gamma_val > 0.0  →  carbon-aware mode
    """
    rng = np.random.default_rng(_SEED)
    agents = [
        IntersectionAgent(
            agent_id=i,
            initial_queue=0.0,
            capacity=_CAPACITY,
            carbon_budget=_CARBON_BUDGET,
            gamma=gamma_val,
            horizon=horizon,
        )
        for i in range(num_agents)
    ]
    demand_streams = rng.poisson(
        lam=_MEAN_DEMAND, size=(num_agents, _STEPS + horizon)
    )

    emissions, avg_queues, green_times = [], [], []
    for t in range(_STEPS):
        step_em, step_q, step_g = 0.0, [], []
        for idx, agent in enumerate(agents):
            forecast = demand_streams[idx, t: t + horizon].astype(float)
            forecast[0] = apply_shock(t, forecast[0], _SHOCK_STEP, shock_factor)
            g = agent.step(forecast)
            step_em += agent.history_emission[-1]
            step_q.append(agent.queue)
            step_g.append(g)
        emissions.append(step_em)
        avg_queues.append(float(np.mean(step_q)))
        green_times.append(step_g)
        # FedAvg every K steps (only relevant when gamma > 0)
        if gamma_val > 0 and (t + 1) % _FEDAVG_INTV == 0:
            weights = [a.get_weights() for a in agents]
            gw = fedavg(weights)
            for a in agents:
                a.set_weights(gw)

    return {"emissions": emissions, "avg_queues": avg_queues, "green_times": green_times}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Traffic Control",
    page_icon="🚦",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏙️ Smart City Low-Carbon Traffic Control Center")
st.markdown("**Distributed MPC with Federated Sustainability Learning**")
st.markdown("---")

# ── About section ─────────────────────────────────────────────────────────────
with st.expander("ℹ️  How it works", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.info(
        "**Multi-Agent System**\n\n"
        "Each intersection runs its own MPC. Agents independently solve "
        "a convex optimisation problem each timestep to choose green-time "
        "fractions that clear queues efficiently."
    )
    c2.info(
        "**Carbon Trade-off**\n\n"
        "The MPC objective balances *idle emissions* (queued vehicles) and "
        "*throughput emissions* (vehicles served). Higher γ shifts priority "
        "toward emission reduction, accepting slightly longer queues."
    )
    c3.info(
        "**Federated γ Synchronisation**\n\n"
        "Every 5 steps, agents share their emission-penalty weight γ via "
        "Federated Averaging. This propagates learned carbon efficiency "
        "across the network without sharing raw traffic data."
    )

st.markdown("---")

# ── Sidebar sliders ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Parameters")
gamma = st.sidebar.slider(
    "Emission penalty γ  (carbon-aware mode)",
    min_value=0.0, max_value=2.0, value=1.5, step=0.1,
    help="γ = 0 → pure baseline. Higher = stronger carbon awareness.",
)
num_agents = st.sidebar.slider(
    "Number of intersection agents", min_value=1, max_value=10, value=4,
)
shock_factor = st.sidebar.slider(
    "Shock demand multiplier  (at step 20)",
    min_value=1.0, max_value=5.0, value=3.0, step=0.5,
)
horizon = st.sidebar.slider(
    "MPC look-ahead horizon (steps)", min_value=3, max_value=10, value=8,
)
st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Run simulation when button pressed ────────────────────────────────────────
if run_btn:
    with st.spinner("Running baseline (γ = 0)…"):
        baseline = _run(0.0, num_agents, horizon, shock_factor)
    with st.spinner(f"Running carbon-aware (γ = {gamma})…"):
        carbon_aware = _run(gamma, num_agents, horizon, shock_factor)

    tb  = sum(baseline["emissions"])
    tc  = sum(carbon_aware["emissions"])
    qb  = float(np.mean(baseline["avg_queues"]))
    qc  = float(np.mean(carbon_aware["avg_queues"]))
    sav = (1 - tc / tb) * 100 if tb > 0 else 0.0
    qdelta = (qc / qb - 1) * 100 if qb > 0 else 0.0

    # ── Metric cards ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔴 Baseline Emission",     f"{tb:,.1f} cu")
    m2.metric("🟢 Carbon-Aware Emission", f"{tc:,.1f} cu",
              delta=f"−{abs(tb - tc):,.1f}",  delta_color="normal")
    m3.metric("♻️ Carbon Savings",        f"{sav:.1f}%")
    m4.metric("🚗 Avg Queue (carbon-aware)", f"{qc:.2f} veh",
              delta=f"{qdelta:+.1f}% vs baseline", delta_color="inverse")

    st.markdown("---")

    # ── Plots ─────────────────────────────────────────────────────────────────
    steps = list(range(_STEPS))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("#0e1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, alpha=0.12, color="white")
        ax.axvline(x=_SHOCK_STEP, color="#f0a500", linewidth=1.4, linestyle="--", alpha=0.85)

    ax1.plot(steps, baseline["emissions"],
             color="#e74c3c", linewidth=1.8, linestyle="--", label="Baseline (γ = 0)")
    ax1.plot(steps, carbon_aware["emissions"],
             color="#2ecc71", linewidth=1.8, label=f"Carbon-Aware (γ = {gamma})")
    ax1.set_ylabel("Total Emission (carbon units)", color="white")
    ax1.set_title("📈  Emissions per Timestep", color="white", fontsize=13)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9, fontsize=10)

    # Shock label on emission axes
    em_top = max(max(baseline["emissions"]), max(carbon_aware["emissions"]))
    ax1.text(_SHOCK_STEP + 0.5, em_top * 0.90, "⚡ Shock",
             color="#f0a500", fontsize=9, va="top")

    ax2.plot(steps, baseline["avg_queues"],
             color="#e74c3c", linewidth=1.8, linestyle="--", label="Baseline (γ = 0)")
    ax2.plot(steps, carbon_aware["avg_queues"],
             color="#2ecc71", linewidth=1.8, label=f"Carbon-Aware (γ = {gamma})")
    ax2.set_xlabel("Timestep", color="white")
    ax2.set_ylabel("Average Queue Length (vehicles)", color="white")
    ax2.set_title("🚗  Average Queue per Timestep", color="white", fontsize=13)
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9, fontsize=10)

    q_top = max(max(baseline["avg_queues"]), max(carbon_aware["avg_queues"]))
    ax2.text(_SHOCK_STEP + 0.5, q_top * 0.90, "⚡ Shock",
             color="#f0a500", fontsize=9, va="top")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close(fig)

    # ── Shock recovery ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Shock Recovery Window  (steps 20 – 29)")
    sc1, sc2 = st.columns(2)
    b_shock = float(np.mean(baseline["avg_queues"][_SHOCK_STEP: _SHOCK_STEP + 10]))
    c_shock = float(np.mean(carbon_aware["avg_queues"][_SHOCK_STEP: _SHOCK_STEP + 10]))
    sc1.metric("Baseline avg queue (shock window)",      f"{b_shock:.2f} veh")
    sc2.metric("Carbon-aware avg queue (shock window)",  f"{c_shock:.2f} veh",
               delta=f"{(c_shock / b_shock - 1) * 100:+.1f}%", delta_color="inverse")

else:
    st.info(
        "👈  Adjust the sliders in the sidebar, then click **▶  Run Simulation**.",
        icon="🚦",
    )
