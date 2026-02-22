"""
Carbon Emission Model
=====================

Models carbon emissions from two sources at an intersection:

1. **Idle emissions** — vehicles waiting in queue burn fuel while idling.
2. **Throughput emissions** — vehicles *served* (passing through) also emit.

Mathematical Model
------------------
    E_idle(t)       = idle_factor × queue(t)
    E_throughput(t) = throughput_factor × capacity × green_time(t)
    E_total(t)      = E_idle(t) + E_throughput(t)

This dual-source model creates a genuine trade-off:
    • Maximising green_time reduces idle emissions (shorter queues)
      but increases throughput emissions (more vehicles served).
    • The carbon-aware controller must balance these competing effects.
"""

import numpy as np


def emission_model(queue, green_time=0.5, capacity=10.0,
                   idle_factor=0.5, throughput_factor=0.3):
    """
    Compute total carbon emissions (idle + throughput).

    Parameters
    ----------
    queue : float or np.ndarray
        Number of vehicles currently in queue (≥ 0).
    green_time : float
        Green-time fraction applied this timestep [0, 1].
    capacity : float
        Maximum vehicles served per full-green timestep.
    idle_factor : float, optional
        Emission per idling vehicle per timestep (default 0.5).
    throughput_factor : float, optional
        Emission per vehicle actually served (default 0.3).

    Returns
    -------
    float or np.ndarray
        Total estimated carbon emissions for this timestep.
    """
    idle_emission = idle_factor * np.asarray(queue, dtype=float)
    throughput_emission = throughput_factor * capacity * green_time
    return idle_emission + throughput_emission
