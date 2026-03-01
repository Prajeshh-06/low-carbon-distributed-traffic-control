"""
Multi-Agent Traffic Simulation
==============================

Orchestrates N intersection agents over T timesteps with:
    • Poisson-distributed demand (or LSTM-predicted demand)
    • Shock events
    • Periodic Federated Averaging synchronisation

Returns a metrics dictionary for downstream analysis and plotting.
"""

import numpy as np

from federated.agent import IntersectionAgent
from federated.fedavg import fedavg
from experiments.shock_events import apply_shock
from core.demand_forecaster import LSTMDemandForecaster


def run_simulation(
    num_agents=4,
    steps=100,
    gamma=0.0,
    federated_sync=False,
    seed=42,
    horizon=5,
    capacity=15.0,
    carbon_budget=15.0,
    mean_demand=7.0,
    fedavg_interval=5,
    shock_step=20,
    shock_factor=3.0,
    use_lstm=False,
):
    """
    Run a full multi-agent traffic simulation.

    Parameters
    ----------
    num_agents : int
        Number of intersection agents.
    steps : int
        Simulation duration in timesteps.
    gamma : float
        Carbon-emission penalty weight for the MPC objective.
    federated_sync : bool
        If True, periodically synchronise γ across agents via FedAvg.
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
    use_lstm : bool
        If True, use LSTM-based demand prediction instead of raw Poisson
        draws for the MPC forecast horizon. The LSTM learns online from
        observed (realised) demand at each timestep.

    Returns
    -------
    dict
        Keys:
            ``total_emission``       — list[float], total emissions per timestep
            ``avg_queue``            — list[float], mean queue across agents
            ``green_time_trajectory``— list[list[float]], green per agent
            ``gamma_trajectory``     — list[list[float]], gamma per agent
            ``forecast_errors``      — list[float], mean LSTM forecast error
                                       (only present when use_lstm=True)
    """
    rng = np.random.default_rng(seed)

    # Create agents
    agents = [
        IntersectionAgent(
            agent_id=i,
            initial_queue=0.0,
            capacity=capacity,
            carbon_budget=carbon_budget,
            gamma=gamma,
            horizon=horizon,
        )
        for i in range(num_agents)
    ]

    # Create per-agent LSTM forecasters (only used when use_lstm=True)
    forecasters = [
        LSTMDemandForecaster(hidden_size=16, lookback=10,
                             learning_rate=0.005, seed=seed + i)
        for i in range(num_agents)
    ] if use_lstm else None

    # Pre-generate demand streams  (Poisson, shape: agents × steps+horizon)
    demand_streams = rng.poisson(lam=mean_demand, size=(num_agents, steps + horizon))

    # Metrics accumulators
    total_emissions = []
    avg_queues = []
    green_times = []
    gamma_trajectories = []
    forecast_errors = [] if use_lstm else None

    for t in range(steps):
        step_emissions = 0.0
        step_queues = []
        step_greens = []
        step_gammas = []
        step_forecast_err = []

        for idx, agent in enumerate(agents):
            # Realised demand for this step (with shock applied)
            realised_demand = apply_shock(
                t, float(demand_streams[idx, t]), shock_step, shock_factor
            )

            if use_lstm and forecasters is not None:
                # Feed the realised demand to the LSTM for online learning
                forecasters[idx].observe(realised_demand)

                # Use LSTM's multi-step prediction as the MPC forecast
                lstm_forecast = forecasters[idx].predict(horizon=horizon)

                # Build forecast: first element is the actual realised demand,
                # remaining elements come from the LSTM prediction
                forecast = np.empty(horizon)
                forecast[0] = realised_demand
                forecast[1:] = lstm_forecast[1:]

                # Track forecast error (how well LSTM predicted this step)
                if len(forecasters[idx].history) >= 2:
                    prev_pred = forecasters[idx].predict(horizon=1)[0]
                    step_forecast_err.append(abs(prev_pred - realised_demand))
            else:
                # Original: use raw Poisson draws for the forecast window
                forecast = demand_streams[idx, t: t + horizon].astype(float)
                forecast[0] = realised_demand

            # Agent step
            g = agent.step(forecast)

            step_emissions += agent.history_emission[-1]
            step_queues.append(agent.queue)
            step_greens.append(g)
            step_gammas.append(agent.local_model_weight)

        total_emissions.append(step_emissions)
        avg_queues.append(float(np.mean(step_queues)))
        green_times.append(step_greens)
        gamma_trajectories.append(step_gammas)

        if use_lstm and forecast_errors is not None:
            avg_err = float(np.mean(step_forecast_err)) if step_forecast_err else 0.0
            forecast_errors.append(avg_err)

        # Periodic Federated Averaging
        if federated_sync and (t + 1) % fedavg_interval == 0:
            weights = [agent.get_weights() for agent in agents]
            global_weight = fedavg(weights)
            for agent in agents:
                agent.set_weights(global_weight)

    result = {
        "total_emission": total_emissions,
        "avg_queue": avg_queues,
        "green_time_trajectory": green_times,
        "gamma_trajectory": gamma_trajectories,
    }
    if use_lstm and forecast_errors is not None:
        result["forecast_errors"] = forecast_errors

    return result
