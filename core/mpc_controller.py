"""
Model Predictive Controller for Traffic Signal Optimization
===========================================================

Solves a finite-horizon convex program to determine the optimal green-time
allocation at an intersection, balancing congestion, delay, and carbon
emissions.

Optimization Formulation
------------------------
Decision variables:
    g[t]  ∈ [0, 1]       green-time fraction for t = 0 … H-1
    q[t]  ≥  0            predicted queue length  for t = 1 … H

Dynamics (inequality relaxation — tight at optimum because q is minimised):
    q[t+1]  ≥  q[t] + d[t] − C · g[t]       ∀ t = 0 … H-1
    q[t+1]  ≥  0                              ∀ t = 0 … H-1

Objective:
    min  Σ_{t=1}^{H}  [ α · q[t]  +  β · q[t]²  +  γ · η · q[t] ]

        α  — congestion weight          (linear queue penalty)
        β  — delay weight               (quadratic queue penalty)
        γ  — carbon / sustainability weight (linear, kept separate for
             interpretability even though it is also linear in q)
        η  — idle emission factor

Solver: OSQP  (problem is a clean QP with linear constraints)
"""

import numpy as np
import cvxpy as cp


def solve_mpc(
    current_queue,
    demand_forecast,
    horizon,
    capacity,
    alpha=0.1,
    beta=0.01,
    gamma=0.0,
    idle_factor=0.2,
    max_green=0.85,
):
    """
    Solve a receding-horizon MPC for one intersection.

    Parameters
    ----------
    current_queue : float
        Current number of queued vehicles (≥ 0).
    demand_forecast : array-like of float, length ``horizon``
        Predicted incoming demand for each step of the horizon.
    horizon : int
        Number of look-ahead steps  (H).
    capacity : float
        Maximum vehicles served per full green timestep.
    alpha : float, optional
        Congestion penalty weight (default 1.0).
    beta : float, optional
        Delay (quadratic queue) penalty weight (default 0.5).
    gamma : float, optional
        Carbon-emission penalty weight (default 0.0 = no carbon awareness).
    idle_factor : float, optional
        Emission per idling vehicle per timestep (default 0.5).

    Returns
    -------
    float
        Optimal green-time fraction for the *current* timestep (g[0]).
        Returns 0.5 as a safe fallback if the solver fails.

    Notes
    -----
    The inequality-based queue dynamics are exact at optimum because the
    objective drives q downward — the constraints bind.  This avoids the
    need for ``cvxpy.maximum`` and keeps the problem a standard QP
    solvable by OSQP.

    The ``max_green`` parameter caps green-time per step to simulate shared
    signal cycles (competing phases at a real intersection).  When
    demand > capacity × max_green, queues build and the carbon term in the
    objective creates a meaningful trade-off.
    """
    demand = np.asarray(demand_forecast, dtype=float)[:horizon]

    # ---- decision variables ------------------------------------------------
    g = cp.Variable(horizon, name="green_time")   # green-time fractions
    q = cp.Variable(horizon, name="queue")         # predicted queue q[1..H]

    # ---- constraints -------------------------------------------------------
    constraints = [
        g >= 0,
        g <= max_green,   # shared cycle — can't use full green every step
        q >= 0,
    ]

    # Queue dynamics: q[t] represents queue at time t+1
    # q[0] >= current_queue + demand[0] - capacity * g[0]
    # q[t] >= q[t-1] + demand[t] - capacity * g[t]   for t >= 1
    constraints.append(q[0] >= current_queue + demand[0] - capacity * g[0])
    for t in range(1, horizon):
        constraints.append(q[t] >= q[t - 1] + demand[t] - capacity * g[t])

    # ---- objective ---------------------------------------------------------
    # Four separate penalty terms for interpretability:
    #   α · q[t]           — congestion cost         (linear in queue)
    #   β · q[t]²          — delay cost              (quadratic in queue)
    #   γ · η · q[t]       — idle emission cost      (from waiting vehicles)
    #   γ · ρ · C · g[t]   — throughput emission cost (from vehicles served)
    #
    # The throughput term is key: serving vehicles (high g) also produces
    # emissions (fuel burned while driving through).  This creates a genuine
    # trade-off — baseline (γ=0) maximises throughput, while carbon-aware
    # mode balances queue clearance against overall emissions.
    throughput_factor = 4.5  # emission per vehicle actually served

    congestion_cost = alpha * cp.sum(q)
    delay_cost = beta * cp.sum_squares(q)
    idle_emission_cost = gamma * idle_factor * cp.sum(q)
    throughput_emission_cost = gamma * throughput_factor * capacity * cp.sum(g)

    objective = cp.Minimize(
        congestion_cost + delay_cost + idle_emission_cost + throughput_emission_cost
    )

    # ---- solve -------------------------------------------------------------
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.SCS, max_iters=1000, verbose=False)
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float(np.clip(g.value[0], 0.0, 1.0))
    except (cp.SolverError, Exception):
        pass  # fall through to safe default

    # Fallback: half-green if solver fails
    return 0.5

