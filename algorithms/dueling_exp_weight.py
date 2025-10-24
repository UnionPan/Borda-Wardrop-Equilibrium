"""
Dueling ExpWeight algorithm for population-based traffic assignment.

Implements a preference-based variant of ExpWeight where each OD pair conducts
pairwise duels between candidate paths, collects estimated Borda scores from the
observed wins, and updates its path distribution accordingly.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from algorithms.base import NonAtomicAlgorithm
from env.traffic_model import AtomicTrafficEnvironment, NonAtomicTrafficEnvironment


class DuelingExpWeight(NonAtomicAlgorithm):
    """
    Preference-based exponential-weights algorithm.

    For every OD pair we maintain empirical win counts between paths. Each
    iteration:
      1. Sample two distinct paths according to the current flow-induced
         distribution.
      2. Compute the preference probability between them using a softmax on the
         negated travel times.
      3. Draw the duel outcome, update bidirectional win/duel counts.
      4. Form estimated Borda scores from the empirical win rates.
      5. Apply an exponentiated-gradient update to the path weights.
    """

    def __init__(self,
                 env: NonAtomicTrafficEnvironment | AtomicTrafficEnvironment,
                 max_iterations: int = 200,
                 learning_rate: float = 0.5,
                 temperature: float = 1.0,
                 duels_per_iteration: int = 1,
                 tolerance: float = 1e-4,
                 random_state: Optional[np.random.Generator] = None):
        """
        Parameters
        ----------
        env : NonAtomicTrafficEnvironment or AtomicTrafficEnvironment
            Traffic environment providing path-time feedback.
        max_iterations : int
            Maximum number of iterations.
        learning_rate : float
            Step size applied to (normalised) Borda scores during weight update.
        temperature : float
            Softmax temperature for preference probabilities. Smaller values
            yield more deterministic preferences.
        duels_per_iteration : int
            Number of independent duels sampled for each OD pair per iteration.
        tolerance : float
            Convergence tolerance on the relative change of path flow.
        random_state : np.random.Generator, optional
            RNG for reproducibility.
        """
        super().__init__(env, max_iterations)

        self.learning_rate = learning_rate
        self.temperature = max(float(temperature), 1e-8)
        self.duels_per_iteration = max(int(duels_per_iteration), 1)
        self.tolerance = tolerance
        self.rng = random_state or np.random.default_rng()

        # Cache OD metadata for fast vectorised access.
        paths_category = self.env.traffic_network.paths_category()
        self.od_path_indices = [
            np.array([i for i, cat in enumerate(paths_category) if cat == od_idx], dtype=int)
            for od_idx in range(self.env.num_od_pairs)
        ]

        # Initialise empirical win/duel counts for every OD block.
        self._wins = []
        self._duels = []
        for indices in self.od_path_indices:
            k = len(indices)
            if k == 0:
                self._wins.append(np.zeros((0, 0)))
                self._duels.append(np.zeros((0, 0)))
            else:
                self._wins.append(np.zeros((k, k), dtype=float))
                self._duels.append(np.zeros((k, k), dtype=float))

        # Initialise positive weights so log-space updates remain stable.
        self.path_weights = np.ones(self.env.num_paths, dtype=float)

        # History trackers specific to dueling.
        self.history['borda_scores'] = []
        self.history['distribution_change'] = []

    # ------------------------------------------------------------------ #
    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        """
        Run duels, update empirical scores, and produce new path flows.
        """
        borda_scores = np.zeros(self.env.num_paths, dtype=float)
        new_weights = self.path_weights.copy()
        new_path_flow = np.zeros(self.env.num_paths, dtype=float)

        for od_idx, indices in enumerate(self.od_path_indices):
            k = len(indices)
            if k == 0:
                continue

            demand = float(self.env.demands[od_idx])
            wins = self._wins[od_idx]
            duels = self._duels[od_idx]

            # Degenerate cases: single path or zero strategic demand.
            if demand <= 0 or k == 1:
                new_path_flow[indices] = 0.0
                continue

            # ------------------------------------------------------------------
            # Sampling two paths per duel (vectorised via cumulative probabilities)
            # ------------------------------------------------------------------
            weights_segment = self.path_weights[indices]
            prob = self._normalise_probabilities(weights_segment)

            # Sample duels_per_iteration opponents without replacement when possible.
            duel_choices = self._sample_duels(prob, k)

            # Preference probabilities from softmax on travel times.
            times = path_time[indices]
            pref_matrix = self._preference_matrix(times)

            # Update empirical wins/duels in batch.
            self._update_empirical_counts(wins, duels, pref_matrix, duel_choices)

            # Estimated win probabilities; guard against zero duels.
            with np.errstate(divide='ignore', invalid='ignore'):
                win_prob = np.true_divide(wins, duels, out=np.zeros_like(wins), where=duels > 0)

            # By definition, ties on identical comparisons count as 0.5.
            np.fill_diagonal(win_prob, 0.5)

            # Estimated Borda score = average win probability over opponents.
            row_sums = win_prob.sum(axis=1) - 0.5  # remove self-comparison.
            borda = row_sums / (k - 1)
            borda_scores[indices] = borda

            # Exponential-weights update with centred, normalised Borda scores.
            centred = borda - np.mean(borda)
            max_abs = np.max(np.abs(centred))
            if max_abs > 0:
                centred /= max_abs
            weights_segment = weights_segment * np.exp(self.learning_rate * centred)
            weights_segment = np.maximum(weights_segment, 1e-12)
            new_weights[indices] = weights_segment

            # Convert weights into flows that respect OD demand.
            prob_updated = self._normalise_probabilities(weights_segment)
            new_path_flow[indices] = demand * prob_updated

        # Record convergence statistics.
        if self.iteration > 0 and self.current_path_flow is not None:
            diff = np.linalg.norm(new_path_flow - self.current_path_flow)
            denom = np.linalg.norm(self.current_path_flow) + 1e-10
            dist_change = diff / denom
        else:
            dist_change = float('inf')

        self.history['distribution_change'].append(dist_change)
        self.history['borda_scores'].append(borda_scores.copy())

        self.path_weights = new_weights
        return new_path_flow

    # ------------------------------------------------------------------ #
    def check_convergence(self) -> bool:
        """Check if the change in path distribution falls below tolerance."""
        if not self.history['distribution_change']:
            return False
        return self.history['distribution_change'][-1] < self.tolerance

    # ------------------------------------------------------------------ #
    # Helper methods                                                     #
    # ------------------------------------------------------------------ #
    def _normalise_probabilities(self, weights: np.ndarray) -> np.ndarray:
        """Convert non-negative weights into a proper probability vector."""
        total = weights.sum()
        if total <= 0:
            return np.ones_like(weights) / weights.size
        return weights / total

    def _sample_duels(self, prob: np.ndarray, k: int) -> np.ndarray:
        """
        Sample path index pairs for duels.

        Returns
        -------
        np.ndarray with shape (duels_per_iteration, 2) containing local indices.
        """
        if k == 2:
            # Deterministic pairing when only two options exist.
            return np.tile(np.array([[0, 1]]), (self.duels_per_iteration, 1))

        # Draw samples with replacement but ensure distinct opponents.
        duel_choices = np.zeros((self.duels_per_iteration, 2), dtype=int)
        for d in range(self.duels_per_iteration):
            first = self.rng.choice(k, p=prob)
            # Remove first path temporarily to avoid duplicates.
            mask = np.ones(k, dtype=bool)
            mask[first] = False
            prob_second = self._normalise_probabilities(prob[mask])
            second_choices = np.arange(k)[mask]
            second = self.rng.choice(second_choices, p=prob_second)
            duel_choices[d, :] = [first, second]
        return duel_choices

    def _preference_matrix(self, times: np.ndarray) -> np.ndarray:
        """
        Compute preference probabilities between every pair of paths.
        """
        scaled = -times / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        denom = exp_scaled[:, None] + exp_scaled[None, :]
        denom = np.where(denom <= 0, 1.0, denom)
        pref = exp_scaled[:, None] / denom
        np.fill_diagonal(pref, 0.5)
        return pref

    def _update_empirical_counts(self,
                                 wins: np.ndarray,
                                 duels: np.ndarray,
                                 pref_matrix: np.ndarray,
                                 duel_choices: np.ndarray) -> None:
        """
        Update win/duel matrices for the given OD pair.
        """
        first = duel_choices[:, 0]
        second = duel_choices[:, 1]
        prefs = pref_matrix[first, second]
        outcomes = (self.rng.random(len(first)) < prefs).astype(float)

        np.add.at(duels, (first, second), 1.0)
        np.add.at(duels, (second, first), 1.0)
        np.add.at(wins, (first, second), outcomes)
        np.add.at(wins, (second, first), 1.0 - outcomes)


__all__ = ["DuelingExpWeight"]
