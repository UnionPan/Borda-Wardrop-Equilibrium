"""
Frank-Wolfe Algorithm for Traffic Assignment

Classic method for computing User Equilibrium (Wardrop Equilibrium).
"""

import numpy as np
from typing import Optional
from algorithms.base import NonAtomicAlgorithm
from env.traffic_model import NonAtomicTrafficEnvironment


class FrankWolfe(NonAtomicAlgorithm):
    """
    Frank-Wolfe Algorithm for User Equilibrium.

    Algorithm:
    1. Initialize with uniform flow distribution
    2. At each iteration:
       a. Find direction: All-or-nothing assignment (shortest paths)
       b. Line search: Find optimal step size
       c. Update: Move towards direction by step size
    3. Repeat until convergence

    Convergence: Relative gap < tolerance
    """

    def __init__(self,
                 env: NonAtomicTrafficEnvironment,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 step_size_method: str = 'optimal'):
        """
        Initialize Frank-Wolfe algorithm.

        Parameters
        ----------
        env : NonAtomicTrafficEnvironment
            Traffic environment
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance (relative gap)
        step_size_method : str
            Step size method: 'optimal' (exact line search) or 'msa' (method of successive averages)
        """
        super().__init__(env, max_iterations)
        self.tolerance = tolerance
        self.step_size_method = step_size_method

        # Additional history for Frank-Wolfe specific metrics
        self.history['relative_gap'] = []
        self.history['step_size'] = []

    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        """
        Frank-Wolfe update: direction finding + line search + update.

        Parameters
        ----------
        path_time : np.ndarray
            Current path travel times

        Returns
        -------
        np.ndarray
            Updated path flow
        """
        # Current path flow
        f_current = self.current_path_flow

        # Direction finding: All-or-nothing assignment (shortest paths)
        f_aon = self.env.get_shortest_path_flow(self.current_link_time)

        # Descent direction
        direction = f_aon - f_current

        # Step size determination
        if self.step_size_method == 'optimal':
            step_size = self._optimal_step_size(f_current, direction)
        elif self.step_size_method == 'msa':
            # Method of Successive Averages: α_k = 1/(k+1)
            step_size = 1.0 / (self.iteration + 1)
        else:
            step_size = 1.0 / (self.iteration + 1)

        # Update
        f_new = f_current + step_size * direction

        # Ensure non-negativity
        f_new = np.maximum(f_new, 0)

        # Compute relative gap for convergence check
        link_flow_current = self.current_link_flow
        link_time_current = self.current_link_time

        # Gap = (current travel time) · (current flow) - (current travel time) · (AON flow)
        # Normalized by (current travel time) · (current flow)
        numerator = np.sum(link_time_current * link_flow_current) - \
                   np.sum(link_time_current * self.env.convert_path_flow_to_link_flow(f_aon))
        denominator = np.sum(link_time_current * link_flow_current)

        relative_gap = numerator / (denominator + 1e-10)

        # Store metrics
        self.history['relative_gap'].append(relative_gap)
        self.history['step_size'].append(step_size)

        return f_new

    def _optimal_step_size(self, f_current: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute optimal step size via line search.

        Minimizes: Beckmann(f + α * d) for α ∈ [0, 1]

        For BPR functions, this can be solved with golden section search
        or simple grid search.

        Parameters
        ----------
        f_current : np.ndarray
            Current path flow
        direction : np.ndarray
            Descent direction

        Returns
        -------
        float
            Optimal step size in [0, 1]
        """
        # Convert to link space
        link_flow_current = self.current_link_flow
        link_direction = self.env.convert_path_flow_to_link_flow(direction)

        # Golden section search
        alpha_min = 0.0
        alpha_max = 1.0
        golden_ratio = (np.sqrt(5) - 1) / 2

        # Initial points
        alpha1 = alpha_min + (1 - golden_ratio) * (alpha_max - alpha_min)
        alpha2 = alpha_min + golden_ratio * (alpha_max - alpha_min)

        f1 = self._beckmann_line(link_flow_current, link_direction, alpha1)
        f2 = self._beckmann_line(link_flow_current, link_direction, alpha2)

        # Iterate
        for _ in range(20):  # Sufficient for good accuracy
            if f1 < f2:
                alpha_max = alpha2
                alpha2 = alpha1
                f2 = f1
                alpha1 = alpha_min + (1 - golden_ratio) * (alpha_max - alpha_min)
                f1 = self._beckmann_line(link_flow_current, link_direction, alpha1)
            else:
                alpha_min = alpha1
                alpha1 = alpha2
                f1 = f2
                alpha2 = alpha_min + golden_ratio * (alpha_max - alpha_min)
                f2 = self._beckmann_line(link_flow_current, link_direction, alpha2)

        # Return midpoint
        return (alpha_min + alpha_max) / 2

    def _beckmann_line(self, link_flow: np.ndarray, direction: np.ndarray, alpha: float) -> float:
        """
        Evaluate Beckmann objective along line: f + α * d

        Parameters
        ----------
        link_flow : np.ndarray
            Current link flow
        direction : np.ndarray
            Direction in link space
        alpha : float
            Step size

        Returns
        -------
        float
            Beckmann objective value
        """
        f = link_flow + alpha * direction
        f = np.maximum(f, 0)  # Ensure non-negativity
        return self.env.compute_beckmann_objective(f)

    def check_convergence(self) -> bool:
        """
        Check convergence based on relative gap.

        Returns
        -------
        bool
            True if relative gap < tolerance
        """
        if len(self.history['relative_gap']) == 0:
            return False

        return self.history['relative_gap'][-1] < self.tolerance

    def plot_convergence(self, figsize=(15, 10), save_path: Optional[str] = None):
        """
        Plot convergence metrics (overrides base to add relative gap).

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

        # Relative gap (convergence criterion)
        ax = axes[1, 0]
        if len(self.history['relative_gap']) > 0:
            ax.semilogy(iterations[1:], self.history['relative_gap'], 'purple', linewidth=2)
            ax.axhline(y=self.tolerance, color='red', linestyle='--',
                      linewidth=2, label=f'Tolerance={self.tolerance}')
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Relative Gap', fontsize=11)
            ax.set_title('Convergence Criterion (Relative Gap)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

        # Step size
        ax = axes[1, 1]
        if len(self.history['step_size']) > 0:
            ax.plot(iterations[1:], self.history['step_size'], 'orange', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=11)
            ax.set_ylabel('Step Size α', fontsize=11)
            ax.set_title(f'Step Size ({self.step_size_method})', fontsize=12, fontweight='bold')
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

        plt.suptitle(f'Frank-Wolfe Algorithm - {self.env.loader.network_name}',
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")

        return fig
