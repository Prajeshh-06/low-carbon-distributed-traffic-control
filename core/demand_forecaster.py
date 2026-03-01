"""
LSTM-Based Demand Forecaster
=============================

A lightweight Long Short-Term Memory (LSTM) network implemented in pure
NumPy for traffic demand prediction.  This module provides online
learning capabilities — the model trains on observed demand history
and predicts future demand for the MPC horizon.

Architecture
------------
    Input  →  LSTM Cell (hidden_size units)  →  Dense Layer  →  Predicted demand

Mathematical Model
------------------
    Forget gate:   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    Input gate:    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    Candidate:     c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    Cell state:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    Output gate:   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
    Hidden state:  h_t = o_t ⊙ tanh(c_t)
    Output:        y_t = W_y · h_t + b_y

The model uses online gradient descent (truncated BPTT with a single
step) so it can adapt in real-time as new demand observations arrive.
"""

import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _tanh(x):
    return np.tanh(x)


class LSTMDemandForecaster:
    """
    Online LSTM demand forecaster for a single intersection.

    Parameters
    ----------
    hidden_size : int
        Number of hidden units in the LSTM cell (default 16).
    lookback : int
        Number of past observations to use as input sequence (default 10).
    learning_rate : float
        Step size for online gradient updates (default 0.005).
    seed : int or None
        Random seed for weight initialisation.

    Usage
    -----
    >>> forecaster = LSTMDemandForecaster(hidden_size=16, lookback=10)
    >>> forecaster.observe(7.0)       # feed each observed demand
    >>> forecast = forecaster.predict(horizon=5)  # get predictions
    """

    def __init__(self, hidden_size=16, lookback=10, learning_rate=0.005, seed=None):
        self.hidden_size = hidden_size
        self.lookback = lookback
        self.lr = learning_rate
        self.history = []

        rng = np.random.default_rng(seed)
        scale = 0.1

        # Combined input size = 1 (demand) + hidden_size
        combined = 1 + hidden_size

        # Gate weights: forget, input, candidate, output
        self.W_f = rng.normal(0, scale, (hidden_size, combined))
        self.b_f = np.ones(hidden_size)  # bias toward remembering
        self.W_i = rng.normal(0, scale, (hidden_size, combined))
        self.b_i = np.zeros(hidden_size)
        self.W_c = rng.normal(0, scale, (hidden_size, combined))
        self.b_c = np.zeros(hidden_size)
        self.W_o = rng.normal(0, scale, (hidden_size, combined))
        self.b_o = np.zeros(hidden_size)

        # Output layer: hidden → 1 (predicted demand)
        self.W_y = rng.normal(0, scale, (1, hidden_size))
        self.b_y = np.zeros(1)

        # Internal state
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)

        # Running statistics for input normalisation
        self._mean = 7.0  # default prior
        self._var = 10.0
        self._count = 0

    def _normalize(self, x):
        """Normalize demand value using running statistics."""
        return (x - self._mean) / (np.sqrt(self._var) + 1e-8)

    def _denormalize(self, x):
        """Reverse normalisation to get actual demand value."""
        return x * (np.sqrt(self._var) + 1e-8) + self._mean

    def _update_stats(self, x):
        """Update running mean and variance (Welford's algorithm)."""
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def _lstm_step(self, x_t, h_prev, c_prev):
        """
        Execute one LSTM cell step.

        Parameters
        ----------
        x_t : float
            Current input (normalised demand).
        h_prev : ndarray
            Previous hidden state.
        c_prev : ndarray
            Previous cell state.

        Returns
        -------
        h_t, c_t, y_t, cache
        """
        x_vec = np.array([x_t])
        combined = np.concatenate([h_prev, x_vec])

        f_t = _sigmoid(self.W_f @ combined + self.b_f)
        i_t = _sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = _tanh(self.W_c @ combined + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = _sigmoid(self.W_o @ combined + self.b_o)
        h_t = o_t * _tanh(c_t)

        y_t = float((self.W_y @ h_t + self.b_y)[0])

        cache = (x_vec, combined, f_t, i_t, c_tilde, o_t, c_prev, h_prev, c_t, h_t)
        return h_t, c_t, y_t, cache

    def _backward_step(self, cache, loss_grad):
        """
        Compute gradients for a single LSTM step (truncated BPTT-1).

        This performs one step of backpropagation through the output
        layer and LSTM gates.
        """
        (x_vec, combined, f_t, i_t, c_tilde, o_t, c_prev, h_prev, c_t, h_t) = cache

        # Output layer gradients
        dW_y = loss_grad * h_t.reshape(1, -1)
        db_y = np.array([loss_grad])
        dh = loss_grad * self.W_y.flatten()

        # Through output gate
        dtanh_c = _tanh(c_t)
        do = dh * dtanh_c
        dc = dh * o_t * (1.0 - dtanh_c ** 2)

        # Through cell state
        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i_t

        # Gate pre-activation gradients
        df_raw = df * f_t * (1.0 - f_t)
        di_raw = di * i_t * (1.0 - i_t)
        dc_raw = dc_tilde * (1.0 - c_tilde ** 2)
        do_raw = do * o_t * (1.0 - o_t)

        # Weight gradients
        comb = combined.reshape(1, -1)
        dW_f = df_raw.reshape(-1, 1) @ comb
        dW_i = di_raw.reshape(-1, 1) @ comb
        dW_c = dc_raw.reshape(-1, 1) @ comb
        dW_o = do_raw.reshape(-1, 1) @ comb

        return {
            'W_f': dW_f, 'b_f': df_raw,
            'W_i': dW_i, 'b_i': di_raw,
            'W_c': dW_c, 'b_c': dc_raw,
            'W_o': dW_o, 'b_o': do_raw,
            'W_y': dW_y, 'b_y': db_y,
        }

    def _apply_gradients(self, grads):
        """Apply gradient descent updates with gradient clipping."""
        max_norm = 1.0
        for key in grads:
            g = grads[key]
            norm = np.linalg.norm(g)
            if norm > max_norm:
                g = g * max_norm / norm
            setattr(self, key, getattr(self, key) - self.lr * g)

    def observe(self, demand_value):
        """
        Record an observed demand value and perform one online training step.

        Parameters
        ----------
        demand_value : float
            The actual demand observed at the current timestep.
        """
        self._update_stats(demand_value)
        self.history.append(float(demand_value))

        # Need at least 2 observations to train (input → target pair)
        if len(self.history) < 2:
            return

        # Online training: use (history[-2] → history[-1]) as a training pair
        x_in = self._normalize(self.history[-2])
        target = self._normalize(self.history[-1])

        h_t, c_t, y_t, cache = self._lstm_step(x_in, self.h, self.c)

        # MSE loss gradient: d/dy (y - target)^2 = 2(y - target)
        loss_grad = 2.0 * (y_t - target)

        grads = self._backward_step(cache, loss_grad)
        self._apply_gradients(grads)

        # Update internal state
        self.h = h_t
        self.c = c_t

    def predict(self, horizon=5):
        """
        Generate demand predictions for the next `horizon` timesteps.

        Uses the LSTM autoregressively — each prediction is fed back
        as input for the next step.

        Parameters
        ----------
        horizon : int
            Number of future timesteps to predict.

        Returns
        -------
        np.ndarray
            Array of predicted demand values (denormalised, clipped ≥ 0).
        """
        if len(self.history) == 0:
            # No history: return mean demand estimate
            return np.full(horizon, self._mean)

        predictions = []
        h, c = self.h.copy(), self.c.copy()
        x_in = self._normalize(self.history[-1])

        for _ in range(horizon):
            h, c, y, _ = self._lstm_step(x_in, h, c)
            pred = self._denormalize(y)
            pred = max(pred, 0.0)  # demand cannot be negative
            predictions.append(pred)
            x_in = self._normalize(pred)

        return np.array(predictions)

    def get_training_loss(self):
        """
        Compute the current prediction error on the most recent observation.

        Returns
        -------
        float
            Squared prediction error, or 0.0 if not enough data.
        """
        if len(self.history) < 2:
            return 0.0
        x_in = self._normalize(self.history[-2])
        _, _, y_t, _ = self._lstm_step(x_in, self.h, self.c)
        target = self._normalize(self.history[-1])
        return float((y_t - target) ** 2)
