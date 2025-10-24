"""
Exponential Weight (EXP3-style) Algorithm for Traffic Assignment

Online learning approach for non-atomic traffic assignment.
"""

import numpy as np
from typing import Optional, Tuple
from algorithms.base import NonAtomicAlgorithm
from env.traffic_model import NonAtomicTrafficEnvironment


class ExpWeight(NonAtomicAlgorithm):
    """
    Adaptive gradient (Adam-style) optimizer for traffic assignment.

    Treats path travel times as gradients of the Beckmann potential and
    performs projected Adam updates on path flows for each OD pair. This
    typically converges faster than multiplicative weights while retaining
    feasibility (non-negative path flows that sum to the OD demand).
    """

    def __init__(self,
                 env: NonAtomicTrafficEnvironment,
                 max_iterations: int = 200,
                 learning_rate: Optional[float] = None,
                 tolerance: float = 1e-4,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize Adam-style ExpWeight algorithm.

        Parameters
        ----------
        env : NonAtomicTrafficEnvironment
            Traffic environment
        max_iterations : int
            Maximum iterations
        learning_rate : float, optional
            Base learning rate. If None, uses adaptive schedule 0.05 / sqrt(t).
        tolerance : float
            Convergence tolerance (change in distribution)
        beta1, beta2 : float
            Adam momentum parameters
        epsilon : float
            Numerical stability constant for Adam denominator
        """
        super().__init__(env, max_iterations)
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam state
        self.m = np.zeros(self.env.num_paths)
        self.v = np.zeros(self.env.num_paths)

        # Adaptive learning rate base when none provided
        # IMPORTANT: Traffic assignment needs MUCH larger learning rates than typical ML
        # because flow magnitudes are in hundreds/thousands, not 0-1 range
        self.default_learning_rate = 10.0  # Increased from 0.05

        # NOTE: od_path_indices is already cached in base class (no need to duplicate)

        # Additional history
        self.history['path_weights'] = []
        self.history['learning_rate'] = []
        self.history['distribution_change'] = []
        self.history['gradient_norm'] = []
        self.history['step_norm'] = []

    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        """
        Adam-style projected gradient update on path flows.

        Parameters
        ----------
        path_time : np.ndarray
            Current path travel times (costs)

        Returns
        -------
        np.ndarray
            Updated path flow distribution
        """
        # Determine (possibly adaptive) learning rate
        # IMPORTANT: Scale learning rate by sqrt(average demand) for better stability
        # Using sqrt instead of linear scaling prevents overly aggressive updates
        avg_demand = self.env.demands.mean() if self.env.demands.mean() > 0 else 1.0
        demand_scale = np.sqrt(avg_demand)

        if self.learning_rate is None:
            # Adaptive schedule: decreases with iterations
            eta = (self.default_learning_rate * demand_scale) / np.sqrt(self.iteration + 1)
        else:
            # Fixed learning rate (scaled by sqrt of problem size for stability)
            eta = self.learning_rate * demand_scale

        # Compute gradient: NEGATIVE path times relative to minimum for each OD pair
        # We negate because path time is a COST (minimize), not reward (maximize)
        # Lower cost paths should get negative gradient -> increase flow (gradient descent)
        # Higher cost paths should get positive gradient -> decrease flow
        grad = np.zeros_like(path_time)

        for od_idx in range(self.env.num_od_pairs):
            path_indices = self.od_path_indices[od_idx]
            if len(path_indices) == 0:
                continue

            od_path_times = path_time[path_indices]
            min_time = od_path_times.min()

            # Gradient = -(excess travel time)
            # Minimum path: grad = 0
            # Longer paths: grad = +(time - min_time) → reduce flow
            # We want: grad = (time - min_time) for gradient descent to work
            grad[path_indices] = od_path_times - min_time

        # Adam first/second moments
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)

        # Bias correction
        t = self.iteration + 1  # adam step count (starts at 1)
        m_hat = self.m / (1.0 - self.beta1 ** t)
        v_hat = self.v / (1.0 - self.beta2 ** t)

        # Compute Adam step and apply projection onto feasible set
        step = eta * m_hat / (np.sqrt(v_hat) + self.epsilon)
        raw_flow = self.current_path_flow - step
        path_flow = self._project_flow(raw_flow)

        # Compute distribution change for convergence check
        if self.iteration > 0:
            prev_flow = self.current_path_flow
            dist_change = np.linalg.norm(path_flow - prev_flow) / (np.linalg.norm(prev_flow) + 1e-10)
            self.history['distribution_change'].append(dist_change)
        else:
            self.history['distribution_change'].append(float('inf'))

        # Store metrics
        if self.store_full_arrays:
            self.history['path_weights'].append(path_flow.copy())
        self.history['learning_rate'].append(eta)
        self.history['gradient_norm'].append(np.linalg.norm(grad))
        self.history['step_norm'].append(np.linalg.norm(step))

        return path_flow

    def _project_flow(self, flow: np.ndarray) -> np.ndarray:
        """
        Project path flows onto feasible simplex for each OD pair.

        Ensures non-negative flows that sum to the OD demand.
        """
        projected = np.zeros_like(flow)

        # FIX: od_path_indices is a dict {od_idx: [path_indices]}
        for od_idx in range(self.env.num_od_pairs):
            indices = self.od_path_indices[od_idx]
            od_flow = flow[indices]
            demand = self.env.demands[od_idx]

            if demand <= 0 or len(indices) == 0:
                # No demand -> zero flow
                projected[indices] = 0.0
                continue

            # Project onto simplex with sum equal to demand
            projected[indices] = self._project_simplex(od_flow, demand)

        return projected

    @staticmethod
    def _project_simplex(values: np.ndarray, target_sum: float) -> np.ndarray:
        """
        Project arbitrary vector onto the simplex {x >= 0, sum(x) = target_sum}.
        """
        if target_sum <= 0:
            return np.zeros_like(values)

        # Sort descending for threshold search (Duchi et al., 2008)
        u = np.sort(values)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.nonzero(u - (cssv - target_sum) / (np.arange(len(u)) + 1) > 0)[0]

        if len(rho_candidates) == 0:
            theta = (cssv[-1] - target_sum) / len(values)
        else:
            rho = rho_candidates[-1]
            theta = (cssv[rho] - target_sum) / (rho + 1)

        projected = np.maximum(values - theta, 0.0)

        # Numerical guard to enforce exact sum
        projection_sum = projected.sum()
        if projection_sum <= 0:
            projected = np.ones_like(values) * (target_sum / len(values))
        else:
            projected *= target_sum / projection_sum

        return projected

    def check_convergence(self) -> bool:
        """
        Check convergence based on Beckmann potential improvement.

        Using distribution change is unreliable because projection can make
        the flow barely change even when far from equilibrium.

        Returns
        -------
        bool
            True if Beckmann improvement < tolerance (relative)
        """
        if len(self.history['beckmann_potential']) < 3:
            return False

        # Check relative improvement in Beckmann over last few iterations
        recent_beckmann = self.history['beckmann_potential'][-3:]
        beckmann_change = abs(recent_beckmann[-1] - recent_beckmann[0])
        relative_change = beckmann_change / (abs(recent_beckmann[0]) + 1e-10)

        return relative_change < self.tolerance

    def plot_convergence(self, figsize=(15, 10), save_path: Optional[str] = None):
        """
        Plot convergence metrics for ExpWeight.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        iterations = self.history['iteration']

        # Beckmann potential
        ax = axes[0, 0]
        ax.plot(iterations, self.history['beckmann_potential'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Beckmann Potential', fontsize=11)
        ax.set_title('User Equilibrium Objective', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # System cost
        ax = axes[0, 1]
        ax.plot(iterations, self.history['system_cost'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Total System Cost', fontsize=11)
        ax.set_title('System Optimum Objective', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Average travel time
        ax = axes[0, 2]
        ax.plot(iterations, self.history['avg_travel_time'], 'g-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Average Travel Time', fontsize=11)
        ax.set_title('Average Travel Time per User', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Distribution change (convergence criterion)
        ax = axes[1, 0]
        if len(self.history['distribution_change']) > 0:
            ax.semilogy(iterations[2:], self.history['distribution_change'][1:], 'purple', linewidth=2)
            ax.axhline(y=self.tolerance, color='red', linestyle='--',
                      linewidth=2, label=f'Tolerance={self.tolerance}')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Distribution Change', fontsize=11)
            ax.set_title('Convergence Criterion (||Δp||)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        # Learning rate
        ax = axes[1, 1]
        if len(self.history['learning_rate']) > 0:
            ax.plot(iterations[1:], self.history['learning_rate'], 'orange', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Learning Rate η', fontsize=11)
            rate_type = 'adaptive' if self.learning_rate is None else 'fixed'
            ax.set_title(f'Learning Rate ({rate_type})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Beckmann improvement
        ax = axes[1, 2]
        if len(self.history['beckmann_potential']) > 1:
            beckmann_arr = np.array(self.history['beckmann_potential'])
            improvement = beckmann_arr[0] - beckmann_arr
            ax.plot(iterations, improvement, 'teal', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Objective Improvement', fontsize=11)
            ax.set_title('Beckmann Reduction from Initial', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'ExpWeight Algorithm - {self.env.loader.network_name}',
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")

        return fig

    def plot_weight_evolution(self, od_pair_idx: int = 0,
                             figsize: Tuple = (12, 6),
                             save_path: Optional[str] = None):
        """
        Plot evolution of path weights for a specific OD pair.

        Parameters
        ----------
        od_pair_idx : int
            Index of OD pair to visualize
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt

        # Find paths for this OD pair
        paths_category = self.env.traffic_network.paths_category()
        path_indices = [i for i, cat in enumerate(paths_category) if cat == od_pair_idx]

        if len(path_indices) == 0:
            print(f"No paths found for OD pair {od_pair_idx}")
            return

        # Extract flow history for these paths (recorded post-update)
        if len(self.history['path_weights']) == 0:
            print("No history recorded yet for path flows.")
            return

        weight_history = np.array(self.history['path_weights'])
        iterations = self.history['iteration']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Align iteration indexing with stored history (skip initial state)
        iter_for_history = iterations[1:len(weight_history)+1]

        # Plot 1: Path flows
        for i, path_idx in enumerate(path_indices):
            weights = weight_history[:, path_idx]
            ax1.plot(iter_for_history, weights, linewidth=2, label=f'Path {i+1}', marker='o', markersize=3)

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Flow', fontsize=12)
        ax1.set_title(f'Path Flow Evolution (OD Pair {od_pair_idx})', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Normalized probabilities
        for i, path_idx in enumerate(path_indices):
            # Compute probability at each iteration
            probs = []
            for iter_idx in range(len(weight_history)):
                od_weights = weight_history[iter_idx, path_indices]
                weight_sum = np.sum(od_weights)
                if weight_sum > 0:
                    prob = weight_history[iter_idx, path_idx] / weight_sum
                else:
                    prob = 1.0 / len(path_indices)
                probs.append(prob)

            ax2.plot(iter_for_history, probs, linewidth=2, label=f'Path {i+1}', marker='o', markersize=3)

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title(f'Path Probabilities (OD Pair {od_pair_idx})', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weight evolution plot saved to {save_path}")

        return fig
