"""
Experiment Runner
=================

Convenience wrapper that runs all six simulation configurations and returns
their results for comparison.  Demand streams are sourced from the real
UCI Metro Interstate Traffic Volume dataset (DOI: 10.24432/C5X60B).
"""

from experiments.simulation import run_simulation
from utils.dataset_loader import load_demand_streams


def run_experiment(num_agents=4, steps=100, seed=42, shock_step=20, **kwargs):
    """
    Execute multiple configurations of simulations for comparative analysis.

    All configurations share the same real-data demand streams so that
    differences in results are attributable solely to the controller
    parameters (gamma, horizon, federated sync, LSTM), not to demand
    variability.

    Returns
    -------
    dict
        Dictionary containing results for each configuration key:
        'baseline', 'carbon', 'high_gamma', 'short_horizon',
        'long_horizon', 'lstm_carbon'.
    """
    # ── Load real traffic data (cached after first call) ──────────────────────
    # Need steps + max_horizon rows per agent; max horizon used is 10.
    max_horizon = 10
    demand_streams = load_demand_streams(
        num_agents=num_agents,
        steps=steps,
        horizon=max_horizon,
        seed=seed,
    )

    def _run(horizon=5, **kw):
        """Helper: slice demand to correct horizon length before passing."""
        needed = steps + horizon
        ds = demand_streams[:, :needed]
        return run_simulation(
            num_agents=num_agents,
            steps=steps,
            seed=seed,
            shock_step=shock_step,
            horizon=horizon,
            demand_streams=ds,
            **kw,
        )

    baseline = _run(
        gamma=0.0,
        federated_sync=False,
        horizon=5,
        **kwargs,
    )

    carbon = _run(
        gamma=0.9,
        federated_sync=True,
        horizon=5,
        **kwargs,
    )

    high_gamma = _run(
        gamma=1.8,
        federated_sync=True,
        horizon=5,
        **kwargs,
    )

    short_horizon = _run(
        gamma=0.6,
        federated_sync=True,
        horizon=3,
        **kwargs,
    )

    long_horizon = _run(
        gamma=1.2,
        federated_sync=True,
        horizon=8,
        **kwargs,
    )

    lstm_carbon = _run(
        gamma=1.05,
        federated_sync=True,
        horizon=6,
        use_lstm=True,
        **kwargs,
    )

    return {
        "baseline":      baseline,
        "carbon":        carbon,
        "high_gamma":    high_gamma,
        "short_horizon": short_horizon,
        "long_horizon":  long_horizon,
        "lstm_carbon":   lstm_carbon,
    }
