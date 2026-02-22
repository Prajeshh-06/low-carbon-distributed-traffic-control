"""
Intersection Agent
==================

Each intersection in the network is controlled by one autonomous agent.
The agent owns a local MPC controller, tracks its queue state, accumulates
carbon emissions, and exposes / accepts a *weight* vector for Federated
Averaging.

The only weight currently federated is **γ** (the emission-penalty
coefficient inside the MPC objective).  This keeps the prototype simple
while demonstrating the FedAvg mechanism — and is the natural parameter
to synchronise because it encodes each agent's learned trade-off between
throughput and sustainability.
"""

import numpy as np

from core.dynamics import traffic_dynamics
from core.emission import emission_model
from core.mpc_controller import solve_mpc


class IntersectionAgent:
    """
    Autonomous controller for a single signalised intersection.

    Parameters
    ----------
    agent_id : int
        Unique identifier for logging / debugging.
    initial_queue : float
        Starting queue length (vehicles).
    capacity : float
        Max vehicles served per full-green timestep.
    carbon_budget : float
        Cumulative emission budget before the agent starts increasing γ.
    alpha : float
        MPC congestion weight.
    beta : float
        MPC delay (quadratic) weight.
    gamma : float
        MPC emission weight — this is the federated parameter.
    horizon : int
        MPC look-ahead steps.
    idle_factor : float
        Emission per idling vehicle per timestep.
    """

    def __init__(
        self,
        agent_id: int,
        initial_queue: float = 0.0,
        capacity: float = 10.0,
        carbon_budget: float = 50.0,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        horizon: int = 5,
        idle_factor: float = 0.5,
        max_green: float = 0.6,
    ):
        self.agent_id = agent_id
        self.queue = float(initial_queue)
        self.capacity = capacity
        self.carbon_budget = carbon_budget
        self.cumulative_emission = 0.0

        # MPC parameters
        self.alpha = alpha
        self.beta = beta
        self.local_model_weight = float(gamma)  # γ — the federated weight
        self.horizon = horizon
        self.idle_factor = idle_factor
        self.max_green = max_green

        # Logging
        self.history_queue = []
        self.history_emission = []
        self.history_green = []

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def step(self, demand_forecast):
        """
        Execute one control cycle.

        1. Solve local MPC  →  green_time
        2. Realise the first demand value (apply dynamics)
        3. Compute & accumulate emissions
        4. Adapt γ if carbon budget exceeded

        Parameters
        ----------
        demand_forecast : array-like of float
            Demand predictions for the next ``self.horizon`` steps.

        Returns
        -------
        float
            The green-time fraction applied this step.
        """
        demand = np.asarray(demand_forecast, dtype=float)

        # 1. Solve MPC
        green_time = solve_mpc(
            current_queue=self.queue,
            demand_forecast=demand,
            horizon=self.horizon,
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.local_model_weight,
            idle_factor=self.idle_factor,
            max_green=self.max_green,
        )

        # 2. Apply dynamics with *realised* demand (first element)
        realised_demand = float(demand[0])
        self.update_queue(realised_demand, green_time)

        # 3. Emissions (include both idle and throughput)
        self.compute_emission(green_time)

        # 4. Adapt γ
        self.adapt_carbon_weight()

        # Log
        self.history_green.append(green_time)

        return green_time

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def update_queue(self, demand, green_time):
        """Update queue using the traffic dynamics model."""
        self.queue = float(
            traffic_dynamics(self.queue, demand, green_time, self.capacity)
        )
        self.history_queue.append(self.queue)

    def compute_emission(self, green_time=0.5):
        """Compute current-step emission (idle + throughput) and add to total."""
        e = float(emission_model(
            self.queue, green_time=green_time,
            capacity=self.capacity, idle_factor=self.idle_factor
        ))
        self.cumulative_emission += e
        self.history_emission.append(e)
        return e

    def adapt_carbon_weight(self):
        """
        If cumulative emissions exceed the carbon budget, increase γ by 10 %
        to make the MPC more emission-averse in future steps.
        """
        if self.cumulative_emission > self.carbon_budget:
            self.local_model_weight *= 1.10

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------
    def get_weights(self):
        """Return the local federated weight (γ) as a numpy array."""
        return np.array([self.local_model_weight])

    def set_weights(self, weights):
        """
        Accept a global federated weight and overwrite the local γ.

        Parameters
        ----------
        weights : array-like
            Single-element array whose first entry is the new γ.
        """
        self.local_model_weight = float(np.asarray(weights).flat[0])
