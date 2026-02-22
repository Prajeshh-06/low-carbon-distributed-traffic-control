"""
Experiment Runner
=================

Convenience wrapper that runs both the baseline and carbon-aware simulations
and returns their results for comparison.
"""

from experiments.simulation import run_simulation


def run_experiment(num_agents=4, steps=50, seed=42, **kwargs):
    """
    Execute baseline and carbon-aware simulations side by side.

    Parameters
    ----------
    num_agents : int
        Number of intersection agents.
    steps : int
        Simulation duration in timesteps.
    seed : int
        Random seed (same seed used for both runs for fair comparison).
    **kwargs
        Forwarded to :func:`run_simulation`.

    Returns
    -------
    tuple[dict, dict]
        ``(baseline_results, carbon_aware_results)``
    """
    baseline = run_simulation(
        num_agents=num_agents,
        steps=steps,
        carbon_aware=False,
        seed=seed,
        **kwargs,
    )

    carbon_aware = run_simulation(
        num_agents=num_agents,
        steps=steps,
        carbon_aware=True,
        seed=seed,
        **kwargs,
    )

    return baseline, carbon_aware
