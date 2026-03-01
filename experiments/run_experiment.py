"""
Experiment Runner
=================

Convenience wrapper that runs both the baseline and carbon-aware simulations
and returns their results for comparison.
"""

from experiments.simulation import run_simulation


def run_experiment(num_agents=4, steps=100, seed=42, shock_step=20, **kwargs):
    """
    Execute multiple configurations of simulations for comparative analysis.

    Returns
    -------
    dict
        Dictionary containing results for each configuration.
    """
    baseline = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=0.0,
        federated_sync=False,
        seed=seed,
        horizon=5,
        shock_step=shock_step,
        **kwargs,
    )

    carbon = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=0.9,
        federated_sync=True,
        seed=seed,
        horizon=5,
        shock_step=shock_step,
        **kwargs,
    )

    high_gamma = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=1.5,
        federated_sync=True,
        seed=seed,
        horizon=5,
        shock_step=shock_step,
        **kwargs,
    )

    short_horizon = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=0.9,
        federated_sync=True,
        seed=seed,
        horizon=3,
        shock_step=shock_step,
        **kwargs,
    )

    long_horizon = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=0.9,
        federated_sync=True,
        seed=seed,
        horizon=10,
        shock_step=shock_step,
        **kwargs,
    )

    # LSTM-enhanced: Carbon-aware MPC with learned demand forecasting
    lstm_carbon = run_simulation(
        num_agents=num_agents,
        steps=steps,
        gamma=0.9,
        federated_sync=True,
        seed=seed,
        horizon=5,
        shock_step=shock_step,
        use_lstm=True,
        **kwargs,
    )

    return {
        "baseline": baseline,
        "carbon": carbon,
        "high_gamma": high_gamma,
        "short_horizon": short_horizon,
        "long_horizon": long_horizon,
        "lstm_carbon": lstm_carbon,
    }
