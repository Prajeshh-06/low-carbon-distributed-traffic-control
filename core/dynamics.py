"""
Traffic Queue Dynamics Module
=============================

Implements discrete-time traffic queue evolution for intersection control.

Mathematical Model
------------------
    x(t+1) = max( x(t) + demand(t) − capacity × green_time(t),  0 )

Where:
    x(t)       : queue length (vehicles waiting) at time t
    demand(t)  : incoming vehicle demand at time t
    green_time : fraction of cycle allocated to green [0, 1]
    capacity   : maximum vehicles that can be served per timestep
"""

import numpy as np


def traffic_dynamics(queue, demand, green_time, capacity):
    """
    Compute next-step queue length using the discrete queue evolution model.

    Parameters
    ----------
    queue : float or np.ndarray
        Current number of vehicles in queue  (≥ 0).
    demand : float or np.ndarray
        Incoming vehicle demand this timestep (≥ 0).
    green_time : float or np.ndarray
        Fraction of cycle that is green, in [0, 1].
    capacity : float
        Maximum vehicles that can depart per full green timestep.

    Returns
    -------
    float or np.ndarray
        Updated queue length, guaranteed non-negative.

    Examples
    --------
    >>> traffic_dynamics(10, 5, 0.5, 12)
    9.0
    >>> traffic_dynamics(2, 1, 1.0, 10)
    0.0
    """
    next_queue = queue + demand - capacity * green_time
    return np.maximum(next_queue, 0.0)
