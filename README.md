# TrafficMind — Low-Carbon Distributed Traffic Signal Control

> Multi-Agent Model Predictive Control with Federated Learning for Smart City Sustainability

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

**TrafficMind** is a research-grade prototype of a decentralized, carbon-aware traffic signal control system. It demonstrates how a network of autonomous intersection agents, each running a local Model Predictive Controller (MPC), can collectively reduce carbon emissions through Federated Learning — without sharing raw traffic data.

### Key Features

| Feature | Detail |
|---|---|
| **Multi-Agent MPC** | Each intersection is an independent agent solving a convex QP every timestep |
| **Dual-Source Emission Model** | Idle emissions (queued vehicles) + throughput emissions (vehicles served) |
| **Federated Averaging** | Agents share only their emission-penalty weight γ via FedAvg every 5 steps |
| **Carbon Budget Adaptation** | Agents self-escalate γ when their cumulative emission budget is exceeded |
| **Shock Event Simulation** | 3× demand spike at step 20 to test system resilience |
| **Streamlit Dashboard** | Interactive smart city control panel |

### Results (default parameters)

| Metric | Baseline (γ=0) | Carbon-Aware (γ=1.5) |
|---|---|---|
| Total Emission | 835.92 cu | 768.50 cu |
| Carbon Savings | — | **8.1 %** |
| Avg Queue Length | 3.20 veh | 3.20 veh |
| Queue Impact | — | 0.0 % |

---

## Project Structure

```
trafficmind/
├── core/
│   ├── dynamics.py            # Discrete queue evolution model
│   ├── emission.py            # Dual-source emission model (idle + throughput)
│   ├── mpc_controller.py      # Convex MPC via CVXPY / SCS solver
│   └── demand_forecaster.py   # Demand prediction utilities
├── federated/
│   ├── agent.py               # IntersectionAgent — local MPC + FedAvg interface
│   └── fedavg.py              # θ_global = (1/N) Σ θ_local
├── experiments/
│   ├── shock_events.py        # Demand surge injection
│   ├── simulation.py          # Multi-agent orchestrator
│   └── run_experiment.py      # Baseline vs carbon-aware runner
├── utils/
│   └── plotting.py            # Matplotlib comparison charts
├── figures/                   # Pre-generated experiment result plots (PDF)
├── dashboard.py               # Streamlit smart city dashboard
├── main.py                    # CLI entry point
├── bare_jrnl.tex              # IEEE journal LaTeX source
├── bare_jrnl.pdf              # Compiled research paper
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.9** or higher
- **pip** (Python package manager)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/Prajeshh-06/low-carbon-distributed-traffic-control.git
cd low-carbon-distributed-traffic-control
```

### 2. Create a Virtual Environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `cvxpy` | Convex optimization (MPC solver) |
| `matplotlib` | Plotting & visualization |
| `streamlit` | Interactive web dashboard |

### 4. Run the CLI Simulation

```bash
python main.py
```

This runs a baseline vs. carbon-aware comparison and prints a results table to the console.

### 5. Launch the Interactive Dashboard

```bash
streamlit run dashboard.py
```

Opens at **http://localhost:8501**. Use the sidebar to adjust:

- **γ (gamma)** — emission penalty weight
- **Number of agents** — intersection count
- **Shock factor** — demand spike multiplier
- **MPC horizon** — look-ahead steps

Click **▶ Run Simulation** to visualize results.

---

## Mathematical Formulation

### Queue Dynamics

```
x(t+1) = max(x(t) + d(t) − C · g(t),  0)
```

### MPC Objective (per agent, horizon H)

```
min  Σ_{t=1}^{H} [ α · q[t]  +  β · q[t]²  +  γ · η · q[t]  +  γ · ρ · C · g[t] ]
```

| Symbol | Meaning |
|---|---|
| α | Congestion weight (linear queue cost) |
| β | Delay weight (quadratic queue cost) |
| γ | Carbon / sustainability weight (federated parameter) |
| η | Idle emission factor |
| ρ | Throughput emission factor |
| C | Intersection capacity |

### Constraints

```
q[t+1]  ≥  q[t] + d[t] − C · g[t]     (queue dynamics, inequality form)
q[t+1]  ≥  0                           (non-negativity)
0  ≤  g[t]  ≤  max_green               (shared signal cycle)
```

### Federated Averaging

```
γ_global = (1/N) Σᵢ γᵢ
```

---

## Configuration

Key parameters in `experiments/simulation.py`:

| Parameter | Default | Effect |
|---|---|---|
| `num_agents` | 4 | Number of intersections |
| `steps` | 50 | Simulation timesteps |
| `horizon` | 8 | MPC look-ahead steps |
| `capacity` | 15.0 | Max vehicles served per green step |
| `mean_demand` | 7.0 | Mean Poisson demand rate |
| `carbon_budget` | 15.0 | Per-agent emission budget before γ escalation |
| `shock_step` | 20 | Timestep of demand shock |
| `shock_factor` | 3.0 | Shock demand multiplier |

---

## Figures

The `figures/` directory contains 12 pre-generated experiment plots:

| Figure | Description |
|---|---|
| `fig1_emission_time` | Emission rate over time |
| `fig2_queue_time` | Queue length over time |
| `fig3_green_time` | Green time allocation over time |
| `fig4_gamma_time` | γ adaptation over time |
| `fig5_cumulative_emission` | Cumulative emissions comparison |
| `fig6_total_emission_bar` | Total emission bar chart |
| `fig7_avg_queue_bar` | Average queue length bar chart |
| `fig8_shock_zoom` | Zoomed shock-event response |
| `fig9_horizon_emission` | Horizon vs emission sensitivity |
| `fig10_gamma_emission` | γ vs emission sensitivity |
| `fig11_shock_factor` | Shock factor analysis |
| `fig12_complexity` | Computational complexity |

---

## Extensibility

The modular design supports future integration with:

- **SUMO** via TraCI (replace `core/dynamics.py` with live simulator calls)
- **Graph-based demand models** (replace Poisson demand streams)
- **Real sensor data** ingestion (swap demand sources in `simulation.py`)
- **Advanced FL algorithms** (replace `fedavg.py` with FedProx, SCAFFOLD, etc.)

---

## License

MIT License — see [LICENSE](LICENSE).
