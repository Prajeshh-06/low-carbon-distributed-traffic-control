"""
Federated Averaging (FedAvg)
============================

Implements the classic Federated Averaging aggregation rule for
synchronising agent parameters across the network.

Mathematical Model
------------------
    θ_global  =  (1 / N)  Σ_{i=1}^{N}  θ_local_i

In this prototype the only federated parameter is the emission-penalty
weight γ, but the implementation is generic over numpy arrays.
"""

import numpy as np


def fedavg(weight_list):
    """
    Compute the Federated Average of a collection of local weight vectors.

    Parameters
    ----------
    weight_list : list of array-like
        Each element is one agent's local weight vector (numpy array).
        All arrays must have the same shape.

    Returns
    -------
    np.ndarray
        Element-wise mean of the input weight vectors.
        Returns ``np.array([0.0])`` if the list is empty.

    Examples
    --------
    >>> fedavg([np.array([0.3]), np.array([0.5]), np.array([0.7])])
    array([0.5])
    >>> fedavg([])
    array([0.])
    """
    if not weight_list:
        return np.array([0.0])

    stacked = np.stack([np.asarray(w, dtype=float) for w in weight_list])
    return np.mean(stacked, axis=0)
