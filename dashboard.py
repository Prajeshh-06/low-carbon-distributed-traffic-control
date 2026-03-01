"""
Smart City Low-Carbon Traffic Control Center
=============================================
Professional Streamlit dashboard for interactive multi-agent MPC simulation
with federated sustainability optimization.

Run with:
    streamlit run dashboard.py
"""

import sys
import time
import pathlib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

# Ensure project root is on the import path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from experiments.simulation import run_simulation

# ══════════════════════════════════════════════════════════════════════════════
# Page configuration
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart City Traffic Control Center",
    page_icon="🚦",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS for premium dark‑themed styling
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }

    /* KPI card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a5a;
        border-radius: 12px;
        padding: 18px 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetric"] label {
        color: #a0aec0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0.9rem 0 0.4rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #2d3748;
    }

    /* Savings badge */
    .savings-green  { color: #48bb78; font-weight: 700; }
    .savings-yellow { color: #ecc94b; font-weight: 700; }
    .savings-red    { color: #fc8181; font-weight: 700; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: dark‑themed matplotlib figure
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG  = "#0e1117"
GRID_CLR = "#ffffff"
SHOCK_CLR = "#f0a500"

COLORS = {
    "baseline": "#e74c3c",
    "carbon":   "#2ecc71",
}

def _dark_ax(ax):
    """Apply dark theme to a matplotlib axes."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, alpha=0.12, color=GRID_CLR)


def _shock_line(ax, shock_step, y_top):
    """Draw a vertical dashed shock indicator."""
    ax.axvline(x=shock_step, color=SHOCK_CLR, linewidth=1.4,
               linestyle="--", alpha=0.85)
    ax.text(shock_step + 0.5, y_top * 0.92, "⚡ Shock",
            color=SHOCK_CLR, fontsize=9, va="top")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>"
    "🏙️ Smart City Low-Carbon Traffic Control Center</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#a0aec0; font-size:1.05rem; "
    "margin-top:0; margin-bottom:0.3rem;'>"
    "Distributed MPC with Federated Sustainability Optimization</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — Configuration Controls
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Configuration Controls")

# Scenario dropdown
scenario = st.sidebar.selectbox(
    "📋 Scenario Preset",
    ["Normal Traffic", "High Congestion", "High Shock", "Aggressive Sustainability"],
    help="Auto-adjusts gamma, shock_factor, and horizon for common scenarios.",
)

SCENARIO_MAP = {
    "Normal Traffic":              {"gamma": 0.9, "shock_factor": 2.0, "horizon": 5},
    "High Congestion":             {"gamma": 0.5, "shock_factor": 4.0, "horizon": 8},
    "High Shock":                  {"gamma": 0.9, "shock_factor": 5.0, "horizon": 5},
    "Aggressive Sustainability":   {"gamma": 2.0, "shock_factor": 2.0, "horizon": 6},
}
defaults = SCENARIO_MAP[scenario]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Parameters")

gamma = st.sidebar.slider(
    "Emission penalty γ", 0.0, 2.0, defaults["gamma"], 0.1,
    help="γ = 0 → pure baseline.  Higher → stronger carbon awareness.",
)
horizon = st.sidebar.slider(
    "MPC look-ahead horizon (H)", 3, 12, defaults["horizon"],
)
num_agents = st.sidebar.slider(
    "Number of intersection agents", 1, 10, 4,
)
shock_factor = st.sidebar.slider(
    "Shock demand multiplier (at step 20)", 1.0, 5.0, defaults["shock_factor"], 0.5,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Toggles")
federated_sync = st.sidebar.checkbox("Enable Federated Synchronization", value=True)
realtime_mode  = st.sidebar.checkbox("Enable Real-Time Simulation", value=False)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Shared simulation constants
# ══════════════════════════════════════════════════════════════════════════════
_STEPS      = 100
_SEED       = 42
_SHOCK_STEP = 20

# ══════════════════════════════════════════════════════════════════════════════
# Main panel — runs on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:

    # ── Run simulations ──────────────────────────────────────────────────────
    with st.spinner("Running baseline simulation (γ = 0) …"):
        baseline = run_simulation(
            num_agents=num_agents, steps=_STEPS, gamma=0.0,
            federated_sync=False, seed=_SEED, horizon=horizon,
            shock_step=_SHOCK_STEP, shock_factor=shock_factor,
        )

    with st.spinner(f"Running carbon-aware simulation (γ = {gamma}) …"):
        carbon = run_simulation(
            num_agents=num_agents, steps=_STEPS, gamma=gamma,
            federated_sync=federated_sync, seed=_SEED, horizon=horizon,
            shock_step=_SHOCK_STEP, shock_factor=shock_factor,
        )

    # ── Compute KPIs ─────────────────────────────────────────────────────────
    tb  = sum(baseline["total_emission"])
    tc  = sum(carbon["total_emission"])
    qb  = float(np.mean(baseline["avg_queue"]))
    qc  = float(np.mean(carbon["avg_queue"]))
    sav = (1 - tc / tb) * 100 if tb > 0 else 0.0

    # Savings colour class
    if sav > 8:
        sav_cls = "savings-green"
    elif sav >= 5:
        sav_cls = "savings-yellow"
    else:
        sav_cls = "savings-red"

    # ══════════════════════════════════════════════════════════════════════════
    # TOP — KPI Metrics
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="section-header">📊 Key Performance Indicators</p>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔴 Baseline Total Emission",     f"{tb:,.1f} cu")
    k2.metric("🟢 Carbon-Aware Total Emission", f"{tc:,.1f} cu",
              delta=f"−{abs(tb - tc):,.1f}", delta_color="normal")
    k3.metric("♻️ Carbon Savings",              f"{sav:.1f} %")
    k4.metric("🚗 Average Queue Length",         f"{qc:.2f} veh",
              delta=f"{((qc/qb)-1)*100:+.1f}% vs baseline" if qb > 0 else "",
              delta_color="inverse")

    # Color‑coded savings badge
    st.markdown(
        f'<p style="text-align:center; font-size: 1.05rem;">'
        f'Carbon savings status: <span class="{sav_cls}">{sav:.1f} %</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MIDDLE — Side‑by‑Side Comparison OR Real‑Time Animation
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">📈 Side-by-Side Configuration Comparison</p>',
        unsafe_allow_html=True,
    )

    if realtime_mode:
        # ── Real‑Time animated mode ──────────────────────────────────────────
        rt_col1, rt_col2 = st.columns(2)
        with rt_col1:
            st.markdown("##### Baseline (γ = 0)")
            ph_em_b = st.empty()
            ph_q_b  = st.empty()
        with rt_col2:
            st.markdown(f"##### Carbon-Aware (γ = {gamma})")
            ph_em_c = st.empty()
            ph_q_c  = st.empty()

        prog = st.progress(0, text="Simulating …")

        for t in range(1, _STEPS + 1):
            pct = t / _STEPS
            prog.progress(pct, text=f"Step {t}/{_STEPS}")

            steps_x = list(range(t))

            # Baseline emission
            fig_b_em, ax_b_em = plt.subplots(figsize=(6, 3))
            fig_b_em.patch.set_facecolor(DARK_BG)
            _dark_ax(ax_b_em)
            ax_b_em.plot(steps_x, baseline["total_emission"][:t],
                         color=COLORS["baseline"], linewidth=1.6)
            if t > _SHOCK_STEP:
                _shock_line(ax_b_em, _SHOCK_STEP,
                            max(baseline["total_emission"][:t]))
            ax_b_em.set_ylabel("Emission")
            ax_b_em.set_title("Emission vs Time", color="white", fontsize=11)
            fig_b_em.tight_layout()
            ph_em_b.pyplot(fig_b_em)
            plt.close(fig_b_em)

            # Baseline queue
            fig_b_q, ax_b_q = plt.subplots(figsize=(6, 3))
            fig_b_q.patch.set_facecolor(DARK_BG)
            _dark_ax(ax_b_q)
            ax_b_q.plot(steps_x, baseline["avg_queue"][:t],
                        color=COLORS["baseline"], linewidth=1.6)
            if t > _SHOCK_STEP:
                _shock_line(ax_b_q, _SHOCK_STEP,
                            max(baseline["avg_queue"][:t]))
            ax_b_q.set_ylabel("Queue")
            ax_b_q.set_title("Queue vs Time", color="white", fontsize=11)
            fig_b_q.tight_layout()
            ph_q_b.pyplot(fig_b_q)
            plt.close(fig_b_q)

            # Carbon emission
            fig_c_em, ax_c_em = plt.subplots(figsize=(6, 3))
            fig_c_em.patch.set_facecolor(DARK_BG)
            _dark_ax(ax_c_em)
            ax_c_em.plot(steps_x, carbon["total_emission"][:t],
                         color=COLORS["carbon"], linewidth=1.6)
            if t > _SHOCK_STEP:
                _shock_line(ax_c_em, _SHOCK_STEP,
                            max(carbon["total_emission"][:t]))
            ax_c_em.set_ylabel("Emission")
            ax_c_em.set_title("Emission vs Time", color="white", fontsize=11)
            fig_c_em.tight_layout()
            ph_em_c.pyplot(fig_c_em)
            plt.close(fig_c_em)

            # Carbon queue
            fig_c_q, ax_c_q = plt.subplots(figsize=(6, 3))
            fig_c_q.patch.set_facecolor(DARK_BG)
            _dark_ax(ax_c_q)
            ax_c_q.plot(steps_x, carbon["avg_queue"][:t],
                        color=COLORS["carbon"], linewidth=1.6)
            if t > _SHOCK_STEP:
                _shock_line(ax_c_q, _SHOCK_STEP,
                            max(carbon["avg_queue"][:t]))
            ax_c_q.set_ylabel("Queue")
            ax_c_q.set_title("Queue vs Time", color="white", fontsize=11)
            fig_c_q.tight_layout()
            ph_q_c.pyplot(fig_c_q)
            plt.close(fig_c_q)

            time.sleep(0.04)

        prog.empty()

    else:
        # ── Static full‑trajectory mode ──────────────────────────────────────
        col_left, col_right = st.columns(2)
        steps_x = list(range(_STEPS))

        with col_left:
            st.markdown("##### Baseline (γ = 0)")

            fig1, ax1 = plt.subplots(figsize=(7, 3.5))
            fig1.patch.set_facecolor(DARK_BG)
            _dark_ax(ax1)
            ax1.plot(steps_x, baseline["total_emission"],
                     color=COLORS["baseline"], linewidth=1.8, label="Baseline")
            _shock_line(ax1, _SHOCK_STEP, max(baseline["total_emission"]))
            ax1.set_ylabel("Emission")
            ax1.set_title("Emission vs Time", color="white", fontsize=11)
            ax1.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9)
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            fig2.patch.set_facecolor(DARK_BG)
            _dark_ax(ax2)
            ax2.plot(steps_x, baseline["avg_queue"],
                     color=COLORS["baseline"], linewidth=1.8, label="Baseline")
            _shock_line(ax2, _SHOCK_STEP, max(baseline["avg_queue"]))
            ax2.set_ylabel("Queue Length")
            ax2.set_title("Queue vs Time", color="white", fontsize=11)
            ax2.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with col_right:
            st.markdown(f"##### Carbon-Aware (γ = {gamma})")

            fig3, ax3 = plt.subplots(figsize=(7, 3.5))
            fig3.patch.set_facecolor(DARK_BG)
            _dark_ax(ax3)
            ax3.plot(steps_x, carbon["total_emission"],
                     color=COLORS["carbon"], linewidth=1.8, label="Carbon-Aware")
            _shock_line(ax3, _SHOCK_STEP, max(carbon["total_emission"]))
            ax3.set_ylabel("Emission")
            ax3.set_title("Emission vs Time", color="white", fontsize=11)
            ax3.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

            fig4, ax4 = plt.subplots(figsize=(7, 3.5))
            fig4.patch.set_facecolor(DARK_BG)
            _dark_ax(ax4)
            ax4.plot(steps_x, carbon["avg_queue"],
                     color=COLORS["carbon"], linewidth=1.8, label="Carbon-Aware")
            _shock_line(ax4, _SHOCK_STEP, max(carbon["avg_queue"]))
            ax4.set_ylabel("Queue Length")
            ax4.set_title("Queue vs Time", color="white", fontsize=11)
            ax4.legend(facecolor="#1a1a2e", labelcolor="white", framealpha=0.9)
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # BOTTOM — Advanced Analytics + System Insight Panel
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-header">🔬 Advanced Analytics</p>',
        unsafe_allow_html=True,
    )

    # ── Radar Performance Chart ──────────────────────────────────────────────
    st.markdown("##### 🕸️ Radar Performance Comparison")

    categories = [
        "Total Emission", "Average Queue", "Shock Recovery",
        "Horizon Length", "Sustainability Weight",
    ]

    # Normalise values to 0‑1 for radar (lower is better for emission/queue)
    b_shock_avg = float(np.mean(baseline["avg_queue"][_SHOCK_STEP:_SHOCK_STEP + 10]))
    c_shock_avg = float(np.mean(carbon["avg_queue"][_SHOCK_STEP:_SHOCK_STEP + 10]))
    max_em = max(tb, tc) or 1
    max_q  = max(qb, qc) or 1
    max_sh = max(b_shock_avg, c_shock_avg) or 1

    base_vals = [
        1 - tb / max_em,          # emission (lower → better → higher score)
        1 - qb / max_q,           # queue
        1 - b_shock_avg / max_sh, # shock recovery
        horizon / 12,             # horizon (same for both)
        0.0 / 2.0,               # sustainability weight baseline=0
    ]
    carb_vals = [
        1 - tc / max_em,
        1 - qc / max_q,
        1 - c_shock_avg / max_sh,
        horizon / 12,
        gamma / 2.0,
    ]

    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * pi for n in range(N_cat)]
    angles += angles[:1]
    base_vals += base_vals[:1]
    carb_vals += carb_vals[:1]

    fig_r, ax_r = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor(DARK_BG)
    ax_r.set_facecolor(DARK_BG)

    ax_r.plot(angles, base_vals, "o-", linewidth=2, color=COLORS["baseline"],
              label="Baseline")
    ax_r.fill(angles, base_vals, alpha=0.15, color=COLORS["baseline"])
    ax_r.plot(angles, carb_vals, "o-", linewidth=2, color=COLORS["carbon"],
              label="Carbon-Aware")
    ax_r.fill(angles, carb_vals, alpha=0.15, color=COLORS["carbon"])

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(categories, color="white", fontsize=9)
    ax_r.tick_params(axis="y", colors="#555")
    ax_r.set_title("Performance Radar", color="white", fontsize=13, pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                facecolor="#1a1a2e", labelcolor="white", framealpha=0.9)
    fig_r.tight_layout()

    rc1, rc2, rc3 = st.columns([1, 2, 1])
    with rc2:
        st.pyplot(fig_r)
    plt.close(fig_r)

    st.markdown("---")

    # ── Shock Recovery Detail ────────────────────────────────────────────────
    st.markdown("##### ⚡ Shock Recovery Window  (steps 20 – 29)")
    sc1, sc2 = st.columns(2)
    sc1.metric("Baseline avg queue (shock window)",     f"{b_shock_avg:.2f} veh")
    sc2.metric("Carbon-aware avg queue (shock window)", f"{c_shock_avg:.2f} veh",
               delta=f"{(c_shock_avg/b_shock_avg - 1)*100:+.1f}%" if b_shock_avg > 0 else "",
               delta_color="inverse")

    st.markdown("---")

    # ── System Mathematical Model ────────────────────────────────────────────
    with st.expander("📐 System Mathematical Model", expanded=False):
        st.markdown("**Queue Dynamics**")
        st.latex(r"q_{t+1} = q_t + d_t - \mu \, g_t")

        st.markdown("**Emission Model**")
        st.latex(r"E_t = \eta \, q_t + \rho \, \mu \, g_t")

        st.markdown("**MPC Objective Function**")
        st.latex(
            r"J = \sum_{k=0}^{H-1} "
            r"\bigl(\alpha \, q_{t+k} + \beta \, q_{t+k}^{2} "
            r"+ \gamma \, E_{t+k}\bigr)"
        )

        st.markdown("**Federated Averaging Update**")
        st.latex(
            r"\gamma^{(k+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_i^{(k)}"
        )

else:
    # ── Landing state ────────────────────────────────────────────────────────
    st.info(
        "👈  Configure parameters in the sidebar, choose a scenario, "
        "then click **▶  Run Simulation**.",
        icon="🚦",
    )

    # Show mathematical model even before running
    with st.expander("📐 System Mathematical Model", expanded=True):
        st.markdown("**Queue Dynamics**")
        st.latex(r"q_{t+1} = q_t + d_t - \mu \, g_t")

        st.markdown("**Emission Model**")
        st.latex(r"E_t = \eta \, q_t + \rho \, \mu \, g_t")

        st.markdown("**MPC Objective Function**")
        st.latex(
            r"J = \sum_{k=0}^{H-1} "
            r"\bigl(\alpha \, q_{t+k} + \beta \, q_{t+k}^{2} "
            r"+ \gamma \, E_{t+k}\bigr)"
        )

        st.markdown("**Federated Averaging Update**")
        st.latex(
            r"\gamma^{(k+1)} = \frac{1}{N} \sum_{i=1}^{N} \gamma_i^{(k)}"
        )
