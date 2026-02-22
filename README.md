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
│   ├── dynamics.py          # Discrete queue evolution:  x(t+1) = max(x(t) + d - C·g, 0)
│   ├── emission.py          # Dual-source emission model (idle + throughput)
│   └── mpc_controller.py    # Convex MPC via CVXPY / SCS solver
├── federated/
│   ├── agent.py             # IntersectionAgent — local MPC + FedAvg interface
│   └── fedavg.py            # θ_global = (1/N) Σ θ_local
├── experiments/
│   ├── shock_events.py      # Demand surge injection
│   ├── simulation.py        # Multi-agent orchestrator
│   └── run_experiment.py    # Baseline vs carbon-aware runner
├── utils/
│   └── plotting.py          # Matplotlib comparison charts
├── dashboard.py             # Streamlit smart city dashboard
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the CLI simulation

```bash
python main.py
```

Prints a comparison table (baseline vs carbon-aware) and saves `comparison_plot.png`.

### 3. Launch the interactive dashboard

```bash
python -m streamlit run dashboard.py
```

Opens at **http://localhost:8501**. Adjust sliders for γ, number of agents, shock factor, and MPC horizon, then click **▶ Run Simulation**.

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

## Extensibility

The modular design supports future integration with:

- **SUMO** via TraCI (replace `core/dynamics.py` with live simulator calls)
- **Graph-based demand models** (replace Poisson demand streams)
- **Real sensor data** ingestion (swap demand sources in `simulation.py`)
- **Advanced FL algorithms** (replace `fedavg.py` with FedProx, SCAFFOLD, etc.)

---

## License

MIT License — see [LICENSE](LICENSE).
