"""
Stochastic Frank-Wolfe dynamics for atomic congestion games.

The algorithm keeps track of the standard Frank-Wolfe (continuous) iterate in
expectation but executes the resulting convex combination by randomly
reallocating discrete agents, mirroring the proportional migration rule.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from algorithms.frank_wolfe import FrankWolfe
from env.traffic_model import AtomicTrafficEnvironment


class DuelingFrankWolfe(FrankWolfe):
    """
    Frank-Wolfe with agent-level Bernoulli migration.

    We maintain an expected strategic path flow (the continuous FW iterate).
    After computing the classic FW descent direction and step size, we update
    this expected flow and then realise it by sampling individual agents that
    switch to the all-or-nothing best-response with probability equal to the
    FW step size.
    """

    def __init__(self,
                 env: AtomicTrafficEnvironment,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 step_size_method: str = 'optimal',
                 random_state: Optional[int | np.random.Generator] = None):
        super().__init__(env, max_iterations, tolerance, step_size_method)

        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)

        self.expected_path_flow: Optional[np.ndarray] = None
        self.path_probabilities: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def _initial_probabilities(self) -> np.ndarray:
        """Uniform probabilities per OD pair."""
        probabilities = np.zeros(self.env.num_paths, dtype=float)
        for od_idx, path_indices in enumerate(self.od_path_indices.values()):
            if not path_indices:
                continue
            probabilities[path_indices] = 1.0 / len(path_indices)
        return probabilities

    def initialize_uniform(self) -> np.ndarray:
        """
        Override to sample discrete strategic flow from uniform probabilities.
        """
        self.expected_path_flow = super().initialize_uniform()
        self.path_probabilities = self.env.distribution_from_flow(self.expected_path_flow)
        return self.env.flow_from_distribution(self.path_probabilities, self.rng)

    # ------------------------------------------------------------------ #
    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        if self.expected_path_flow is None:
            raise RuntimeError("Expected flow not initialised. Call initialize_uniform first.")

        f_expected = self.expected_path_flow
        f_aon = self.env.get_shortest_path_flow(self.current_link_time)

        direction = f_aon - f_expected

        if self.step_size_method == 'optimal':
            step_size = self._optimal_step_size(f_expected, direction)
        elif self.step_size_method == 'msa':
            step_size = 1.0 / (self.iteration + 1)
        else:
            step_size = 1.0 / (self.iteration + 1)

        expected_new = f_expected + step_size * direction
        expected_new = np.maximum(expected_new, 0.0)

        self.expected_path_flow = expected_new
        self.path_probabilities = self.env.distribution_from_flow(expected_new)

        link_flow_current = self.current_link_flow
        link_time_current = self.current_link_time
        aon_link_flow = self.env.convert_path_flow_to_link_flow(f_aon)

        numerator = np.sum(link_time_current * link_flow_current) - \
            np.sum(link_time_current * aon_link_flow)
        denominator = np.sum(link_time_current * link_flow_current)
        relative_gap = numerator / (denominator + 1e-10)

        self.history['relative_gap'].append(relative_gap)
        self.history['step_size'].append(step_size)

        return self.env.flow_from_distribution(self.path_probabilities, self.rng)

    # ------------------------------------------------------------------ #
    def check_convergence(self) -> bool:
        """Same convergence test as classic Frank-Wolfe (relative gap)."""
        if not self.history['relative_gap']:
            return False
        return self.history['relative_gap'][-1] < self.tolerance


__all__ = ["DuelingFrankWolfe"]
