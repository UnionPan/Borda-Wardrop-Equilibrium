"""
System Optimum Algorithm for Traffic Assignment

Frank-Wolfe with marginal cost (regularized gradient) for social optimization.
"""

import numpy as np
from typing import Optional
from algorithms.frank_wolfe import FrankWolfe
from env.traffic_model import NonAtomicTrafficEnvironment


class SystemOptimum(FrankWolfe):
    """
    System Optimum Algorithm using Frank-Wolfe with Marginal Costs.

    The System Optimum minimizes total system cost rather than individual costs.
    This is achieved by using marginal costs instead of actual travel times.

    For BPR function: t(f) = t0 * (1 + α * (f/c)^β)
    
    Marginal cost: MC(f) = t(f) + f * t'(f)
                         = t(f) + f * t0 * α * β * (f/c)^β / c
                         = t0 * (1 + α * (f/c)^β) + f * t0 * α * β * (f/c)^β / c
                         = t0 * (1 + α * (1+β) * (f/c)^β)

    Intuition: Marginal cost accounts for the congestion externality each driver
    imposes on all other drivers using the same link.
    """

    def __init__(self,
                 env: NonAtomicTrafficEnvironment,
                 max_iterations: int = 100,
                 tolerance: float = 1e-4,
                 step_size_method: str = 'optimal'):
        """
        Initialize System Optimum algorithm.

        Parameters
        ----------
        env : NonAtomicTrafficEnvironment
            Traffic environment
        max_iterations : int
            Maximum iterations
        tolerance : float
            Convergence tolerance
        step_size_method : str
            Step size method: 'optimal' or 'msa'
        """
        super().__init__(env, max_iterations, tolerance, step_size_method)

        # Add marginal cost history
        self.history['marginal_cost'] = []

    def compute_marginal_cost(self, link_flow: np.ndarray) -> np.ndarray:
        """
        Compute marginal cost for each link.

        MC(f) = t(f) + f * t'(f)

        For BPR: t(f) = t0 * (1 + α * (f/c)^β)
                 t'(f) = t0 * α * β * (f/c)^(β-1) / c

        Therefore: MC(f) = t0 * (1 + α*(f/c)^β) + f * t0 * α * β * (f/c)^(β-1) / c
                         = t0 * (1 + α*(f/c)^β + α*β*(f/c)^β)
                         = t0 * (1 + α*(1+β)*(f/c)^β)

        Parameters
        ----------
        link_flow : np.ndarray
            Link flows

        Returns
        -------
        np.ndarray
            Marginal cost for each link
        """
        # BPR parameters
        t0 = self.env.free_flow_times
        alpha = self.env.alpha
        beta = self.env.beta
        capacity = self.env.capacities

        # Marginal cost = t0 * (1 + α*(1+β)*(f/c)^β)
        marginal_cost = t0 * (1.0 + alpha * (1 + beta) * np.power(link_flow / capacity, beta))

        return marginal_cost

    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        """
        System Optimum update using marginal costs.

        Same as Frank-Wolfe but uses marginal cost for direction finding.

        Parameters
        ----------
        path_time : np.ndarray
            Current path travel times (not used, we compute marginal cost)

        Returns
        -------
        np.ndarray
            Updated path flow
        """
        # Current path flow
        f_current = self.current_path_flow

        # Compute marginal costs
        marginal_link_cost = self.compute_marginal_cost(self.current_link_flow)

        # Compute marginal path costs
        marginal_path_cost = self.env.compute_path_travel_time(marginal_link_cost)

        # Store marginal costs (only if storing full arrays)
        if self.store_full_arrays:
            self.history['marginal_cost'].append(marginal_link_cost.copy())

        # Direction finding: All-or-nothing assignment using MARGINAL costs
        # (This is the key difference from User Equilibrium)
        path_flow_aon = np.zeros(self.env.num_paths)

        for od_idx in range(self.env.num_od_pairs):
            # Use cached path indices (PERFORMANCE FIX)
            path_indices = self.od_path_indices[od_idx]

            # Find path with minimum MARGINAL cost
            od_marginal_costs = marginal_path_cost[path_indices]
            min_idx = np.argmin(od_marginal_costs)
            shortest_path_idx = path_indices[min_idx]

            # Assign all demand to minimum marginal cost path
            path_flow_aon[shortest_path_idx] = self.env.demands[od_idx]

        # Descent direction
        direction = path_flow_aon - f_current

        # Step size determination (same as Frank-Wolfe)
        if self.step_size_method == 'optimal':
            step_size = self._optimal_step_size_system(f_current, direction)
        elif self.step_size_method == 'msa':
            step_size = 1.0 / (self.iteration + 1)
        else:
            step_size = 1.0 / (self.iteration + 1)

        # Update
        f_new = f_current + step_size * direction
        f_new = np.maximum(f_new, 0)

        # Compute relative gap using marginal costs (not regular costs!)
        link_flow_current = self.current_link_flow
        link_flow_aon = self.env.convert_path_flow_to_link_flow(path_flow_aon)

        numerator = np.sum(marginal_link_cost * link_flow_current) - \
                   np.sum(marginal_link_cost * link_flow_aon)
        denominator = np.sum(marginal_link_cost * link_flow_current)

        relative_gap = numerator / (denominator + 1e-10)

        # Store metrics
        self.history['relative_gap'].append(relative_gap)
        self.history['step_size'].append(step_size)

        return f_new

    def _optimal_step_size_system(self, f_current: np.ndarray, direction: np.ndarray) -> float:
        """
        Compute optimal step size for SYSTEM objective (not Beckmann).

        Minimizes: SystemCost(f + α * d) for α ∈ [0, 1]

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

        f1 = self._system_cost_line(link_flow_current, link_direction, alpha1)
        f2 = self._system_cost_line(link_flow_current, link_direction, alpha2)

        # Iterate
        for _ in range(20):
            if f1 < f2:
                alpha_max = alpha2
                alpha2 = alpha1
                f2 = f1
                alpha1 = alpha_min + (1 - golden_ratio) * (alpha_max - alpha_min)
                f1 = self._system_cost_line(link_flow_current, link_direction, alpha1)
            else:
                alpha_min = alpha1
                alpha1 = alpha2
                f1 = f2
                alpha2 = alpha_min + golden_ratio * (alpha_max - alpha_min)
                f2 = self._system_cost_line(link_flow_current, link_direction, alpha2)

        return (alpha_min + alpha_max) / 2

    def _system_cost_line(self, link_flow: np.ndarray, direction: np.ndarray, alpha: float) -> float:
        """
        Evaluate SYSTEM COST along line: f + α * d

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
            System cost value
        """
        f = link_flow + alpha * direction
        f = np.maximum(f, 0)
        return self.env.compute_system_cost(f)

    def plot_convergence(self, figsize=(15, 10), save_path: Optional[str] = None):
        """
        Plot convergence metrics for System Optimum.

        Highlights that we're minimizing SYSTEM cost, not Beckmann.
        """
        fig = super().plot_convergence(figsize, save_path=None)

        # Update title
        fig.suptitle(f'System Optimum Algorithm - {self.env.loader.network_name}',
                    fontsize=16, fontweight='bold', y=0.998)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")

        return fig
