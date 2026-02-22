"""
Shock Events
=============

Simulates unexpected traffic demand surges (e.g., stadium exits, accidents,
flash flooding detours) that stress-test the control system's adaptability.
"""


def apply_shock(step, base_demand, shock_step=20, factor=3.0):
    """
    Conditionally multiply demand by a shock factor at a specific timestep.

    Parameters
    ----------
    step : int
        Current simulation timestep.
    base_demand : float
        Nominal demand before any shock.
    shock_step : int, optional
        Timestep at which the shock occurs (default 20).
    factor : float, optional
        Multiplicative demand surge factor (default 3.0).

    Returns
    -------
    float
        Demand value — shocked or unshocked.

    Examples
    --------
    >>> apply_shock(20, 5.0)
    15.0
    >>> apply_shock(10, 5.0)
    5.0
    """
    if step == shock_step:
        return base_demand * factor
    return base_demand
