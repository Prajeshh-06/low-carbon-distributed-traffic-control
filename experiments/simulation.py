"""
Multi-Agent Traffic Simulation
==============================

Orchestrates N intersection agents over T timesteps with:
    • Poisson-distributed demand
    • Shock events
    • Periodic Federated Averaging synchronisation

Returns a metrics dictionary for downstream analysis and plotting.
"""

import numpy as np

from federated.agent import IntersectionAgent
from federated.fedavg import fedavg
from experiments.shock_events import apply_shock


def run_simulation(
    num_agents=4,
    steps=50,
    carbon_aware=True,
    seed=42,
    horizon=8,
    capacity=15.0,
    carbon_budget=15.0,
    mean_demand=7.0,
    fedavg_interval=5,
    shock_step=20,
    shock_factor=3.0,
):
    """
    Run a full multi-agent traffic simulation.

    Parameters
    ----------
    num_agents : int
        Number of intersection agents.
    steps : int
        Simulation duration in timesteps.
    carbon_aware : bool
        If True, agents start with γ = 0.3 (emission-aware MPC).
        If False, γ = 0.0 (baseline, throughput-only).
    seed : int
        Random seed for reproducibility.
    horizon : int
        MPC look-ahead horizon per agent.
    capacity : float
        Max vehicles served per full-green timestep.
    carbon_budget : float
        Each agent's cumulative emission budget.
    mean_demand : float
        Mean of the Poisson demand distribution.
    fedavg_interval : int
        Perform FedAvg every K timesteps.
    shock_step : int
        Timestep at which the demand shock occurs.
    shock_factor : float
        Demand multiplier during the shock.

    Returns
    -------
    dict
        Keys:
            ``emissions``    — list[float], total emissions per timestep
            ``avg_queues``   — list[float], mean queue across agents per step
            ``green_times``  — list[list[float]], green-time per agent per step
    """
    rng = np.random.default_rng(seed)

    # Initial γ depends on mode
    gamma_init = 1.5 if carbon_aware else 0.0

    # Create agents
    agents = [
        IntersectionAgent(
            agent_id=i,
            initial_queue=0.0,
            capacity=capacity,
            carbon_budget=carbon_budget,
            gamma=gamma_init,
            horizon=horizon,
        )
        for i in range(num_agents)
    ]

    # Pre-generate demand streams  (Poisson, shape: agents × steps+horizon)
    demand_streams = rng.poisson(lam=mean_demand, size=(num_agents, steps + horizon))

    # Metrics accumulators
    total_emissions = []
    avg_queues = []
    green_times = []

    for t in range(steps):
        step_emissions = 0.0
        step_queues = []
        step_greens = []

        for idx, agent in enumerate(agents):
            # Build demand forecast (horizon-length window from t)
            forecast = demand_streams[idx, t: t + horizon].astype(float)

            # Apply shock to the *realised* (first) demand element
            forecast[0] = apply_shock(t, forecast[0], shock_step, shock_factor)

            # Agent step
            g = agent.step(forecast)

            step_emissions += agent.history_emission[-1]
            step_queues.append(agent.queue)
            step_greens.append(g)

        total_emissions.append(step_emissions)
        avg_queues.append(float(np.mean(step_queues)))
        green_times.append(step_greens)

        # Periodic Federated Averaging
        if carbon_aware and (t + 1) % fedavg_interval == 0:
            weights = [agent.get_weights() for agent in agents]
            global_weight = fedavg(weights)
            for agent in agents:
                agent.set_weights(global_weight)

    return {
        "emissions": total_emissions,
        "avg_queues": avg_queues,
        "green_times": green_times,
    }
