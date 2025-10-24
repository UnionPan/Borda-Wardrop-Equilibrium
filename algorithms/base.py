"""
Base Algorithm Classes for Traffic Assignment

Provides base classes for non-atomic traffic assignment algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from env.traffic_model import NonAtomicTrafficEnvironment


class NonAtomicAlgorithm(ABC):
    """
    Base class for non-atomic traffic assignment algorithms.

    Handles:
    - Interaction with NonAtomicTrafficEnvironment
    - Initialization from uniform distribution
    - Data storage across iterations
    - Plotting and visualization

    Subclasses implement specific update schemes (Frank-Wolfe, ExpWeight, etc.)
    """

    def __init__(self, env: NonAtomicTrafficEnvironment, max_iterations: int = 100):
        """
        Initialize non-atomic algorithm.

        Parameters
        ----------
        env : NonAtomicTrafficEnvironment
            Traffic environment to optimize
        max_iterations : int
            Maximum number of iterations
        """
        self.env = env
        self.max_iterations = max_iterations

        # Current state
        self.current_path_flow = None
        self.current_link_flow = None
        self.current_path_time = None
        self.current_link_time = None

        # History storage (OPTIMIZED: only store scalars, not full arrays)
        self.history = {
            'path_flow': [],           # Path flow distribution each iteration (OPTIONAL)
            'link_flow': [],           # Link flow each iteration (OPTIONAL)
            'path_time': [],           # Path travel times each iteration (OPTIONAL)
            'link_time': [],           # Link travel times each iteration (OPTIONAL)
            'beckmann_potential': [],  # Beckmann objective (UE)
            'system_cost': [],         # Total system cost
            'avg_travel_time': [],     # Average travel time
            'iteration': []            # Iteration number
        }

        # Control what to store (to save memory and time)
        self.store_full_arrays = False  # Set to True if you need full history

        # PERFORMANCE: Cache paths_category and OD path indices (computed once!)
        self.paths_category = self.env.traffic_network.paths_category()
        self.od_path_indices = {}  # Maps od_idx -> list of path indices
        for od_idx in range(self.env.num_od_pairs):
            self.od_path_indices[od_idx] = [i for i, cat in enumerate(self.paths_category) if cat == od_idx]

        self.iteration = 0
        self.converged = False

    def initialize_uniform(self) -> np.ndarray:
        """
        Initialize path flow from uniform distribution over paths for each OD pair.

        Returns
        -------
        np.ndarray
            Initial path flow distribution
        """
        path_flow = np.zeros(self.env.num_paths)

        for od_idx in range(self.env.num_od_pairs):
            # Use cached path indices (PERFORMANCE FIX)
            path_indices = self.od_path_indices[od_idx]
            n_paths = len(path_indices)

            # Distribute demand uniformly across paths
            for idx in path_indices:
                path_flow[idx] = self.env.demands[od_idx] / n_paths

        return path_flow

    def step_environment(self, path_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Send path flow to environment and get feedback.

        Parameters
        ----------
        path_flow : np.ndarray
            Path flow distribution to evaluate

        Returns
        -------
        link_flow : np.ndarray
            Resulting link flows
        link_time : np.ndarray
            Link travel times
        path_time : np.ndarray
            Path travel times
        """
        link_flow, link_time, path_time = self.env.step(path_flow)

        # Update current state
        self.current_path_flow = path_flow.copy()
        self.current_link_flow = link_flow.copy()
        self.current_path_time = path_time.copy()
        self.current_link_time = link_time.copy()

        return link_flow, link_time, path_time

    def store_iteration_data(self):
        """Store current state in history (OPTIMIZED: only stores arrays if requested)."""
        self.history['iteration'].append(self.iteration)

        # PERFORMANCE FIX: Only store full arrays if explicitly requested
        if self.store_full_arrays:
            self.history['path_flow'].append(self.current_path_flow.copy())
            self.history['link_flow'].append(self.current_link_flow.copy())
            self.history['path_time'].append(self.current_path_time.copy())
            self.history['link_time'].append(self.current_link_time.copy())

        # Always compute and store objectives (scalars only - fast!)
        beckmann = self.env.compute_beckmann_objective(self.current_link_flow)
        system_cost = self.env.compute_system_cost(self.current_link_flow)
        avg_time = self.env.get_average_travel_time()

        self.history['beckmann_potential'].append(beckmann)
        self.history['system_cost'].append(system_cost)
        self.history['avg_travel_time'].append(avg_time)

    @abstractmethod
    def update_strategy(self, path_time: np.ndarray) -> np.ndarray:
        """
        Update path flow distribution based on feedback.

        This is algorithm-specific and must be implemented by subclasses.

        Parameters
        ----------
        path_time : np.ndarray
            Current path travel times

        Returns
        -------
        np.ndarray
            Updated path flow distribution
        """
        pass

    @abstractmethod
    def check_convergence(self) -> bool:
        """
        Check if algorithm has converged.

        This is algorithm-specific and must be implemented by subclasses.

        Returns
        -------
        bool
            True if converged, False otherwise
        """
        pass

    def run(self, verbose: bool = True) -> Dict:
        """
        Run the algorithm to convergence or max iterations.

        Parameters
        ----------
        verbose : bool
            Print progress information

        Returns
        -------
        Dict
            Final results and history
        """
        # Initialize
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running {self.__class__.__name__}")
            print(f"{'='*70}")
            print(f"Network: {self.env.loader.network_name}")
            print(f"Nodes: {self.env.nx_graph.number_of_nodes()}, "
                  f"Links: {self.env.num_links}, "
                  f"Paths: {self.env.num_paths}")
            print(f"OD Pairs: {self.env.num_od_pairs}, "
                  f"Total Demand: {self.env.demands.sum():.0f}")
            print(f"Max Iterations: {self.max_iterations}\n")

        # Start with uniform distribution
        path_flow = self.initialize_uniform()

        # Initial evaluation
        link_flow, link_time, path_time = self.step_environment(path_flow)
        self.store_iteration_data()

        if verbose:
            print(f"Iter {self.iteration:4d}: "
                  f"Beckmann={self.history['beckmann_potential'][-1]:.2e}, "
                  f"AvgTime={self.history['avg_travel_time'][-1]:.2f}")

        # Main loop
        for self.iteration in range(1, self.max_iterations):
            # Update strategy (algorithm-specific)
            path_flow = self.update_strategy(path_time)

            # Evaluate new strategy
            link_flow, link_time, path_time = self.step_environment(path_flow)
            self.store_iteration_data()

            if verbose and self.iteration % 10 == 0:
                print(f"Iter {self.iteration:4d}: "
                      f"Beckmann={self.history['beckmann_potential'][-1]:.2e}, "
                      f"AvgTime={self.history['avg_travel_time'][-1]:.2f}")

            # Check convergence
            if self.check_convergence():
                self.converged = True
                if verbose:
                    print(f"\n✓ Converged at iteration {self.iteration}")
                break

        if verbose:
            if not self.converged:
                print(f"\n✗ Did not converge within {self.max_iterations} iterations")
            print(f"\nFinal Results:")
            print(f"  Beckmann Objective: {self.history['beckmann_potential'][-1]:.6e}")
            print(f"  System Cost: {self.history['system_cost'][-1]:.6e}")
            print(f"  Avg Travel Time: {self.history['avg_travel_time'][-1]:.4f}")
            print(f"{'='*70}\n")

        return {
            'converged': self.converged,
            'iterations': self.iteration,
            'final_path_flow': self.current_path_flow,
            'final_link_flow': self.current_link_flow,
            'final_beckmann': self.history['beckmann_potential'][-1],
            'final_system_cost': self.history['system_cost'][-1],
            'final_avg_time': self.history['avg_travel_time'][-1],
            'history': self.history
        }

    def plot_convergence(self, figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None):
        """
        Plot convergence metrics over iterations.

        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        iterations = self.history['iteration']

        # Beckmann potential
        ax = axes[0, 0]
        ax.plot(iterations, self.history['beckmann_potential'], 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Beckmann Potential', fontsize=12)
        ax.set_title('User Equilibrium Objective', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # System cost
        ax = axes[0, 1]
        ax.plot(iterations, self.history['system_cost'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Total System Cost', fontsize=12)
        ax.set_title('System Optimum Objective', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Average travel time
        ax = axes[1, 0]
        ax.plot(iterations, self.history['avg_travel_time'], 'g-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average Travel Time', fontsize=12)
        ax.set_title('Average Travel Time per User', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Relative change in Beckmann (convergence indicator)
        ax = axes[1, 1]
        if len(self.history['beckmann_potential']) > 1:
            beckmann_arr = np.array(self.history['beckmann_potential'])
            rel_change = np.abs(np.diff(beckmann_arr) / (beckmann_arr[:-1] + 1e-10))
            ax.semilogy(iterations[1:], rel_change, 'purple', linewidth=2)
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Relative Change', fontsize=12)
            ax.set_title('Convergence Rate (|ΔBeckmann| / Beckmann)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'{self.__class__.__name__} - {self.env.loader.network_name}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")

        return fig

    def plot_link_flows(self, iterations: Optional[List[int]] = None,
                       figsize: Tuple[int, int] = (18, 12),
                       save_path: Optional[str] = None):
        """
        Plot link flow distributions at specific iterations.

        Parameters
        ----------
        iterations : List[int], optional
            Iteration numbers to plot. If None, plots first, middle, last.
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if not self.store_full_arrays or len(self.history['link_flow']) == 0:
            print("Warning: Link flow history not stored. Set store_full_arrays=True to enable.")
            return None

        if iterations is None:
            # Default: first, middle, last
            total_iters = len(self.history['iteration'])
            iterations = [0, total_iters // 2, total_iters - 1]

        n_plots = len(iterations)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        for idx, iter_num in enumerate(iterations):
            link_flow = self.history['link_flow'][iter_num]

            ax = axes[idx]
            ax.bar(range(len(link_flow)), link_flow, alpha=0.7, color='steelblue')
            ax.set_xlabel('Link Index', fontsize=12)
            ax.set_ylabel('Flow', fontsize=12)
            ax.set_title(f'Iteration {iter_num}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Link Flow Evolution - {self.__class__.__name__}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Link flow plot saved to {save_path}")

        return fig

    def visualize_flow_on_map(self, iteration: Optional[int] = None,
                             figsize: Tuple[int, int] = (16, 12),
                             save_path: Optional[str] = None):
        """
        Visualize link flows on geographic map at specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number to visualize. If None, uses current state.
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if iteration is None:
            # Use current link flow
            link_flow = self.current_link_flow
            title = f"{self.__class__.__name__} - Final State"
        else:
            if not self.store_full_arrays or len(self.history['link_flow']) == 0:
                print("Warning: Link flow history not stored. Showing current state instead.")
                link_flow = self.current_link_flow
                title = f"{self.__class__.__name__} - Current State"
            else:
                link_flow = self.history['link_flow'][iteration]
                title = f"{self.__class__.__name__} - Iteration {iteration}"

        self.env.visualize(
            figsize=figsize,
            show_background=True,
            flow_data=link_flow,
            title=title,
            save_path=save_path
        )

    def get_statistics(self) -> Dict:
        """
        Get summary statistics of the optimization.

        Returns
        -------
        Dict
            Summary statistics
        """
        return {
            'algorithm': self.__class__.__name__,
            'network': self.env.loader.network_name,
            'converged': self.converged,
            'iterations': self.iteration,
            'initial_beckmann': self.history['beckmann_potential'][0],
            'final_beckmann': self.history['beckmann_potential'][-1],
            'beckmann_improvement': (self.history['beckmann_potential'][0] -
                                    self.history['beckmann_potential'][-1]),
            'initial_avg_time': self.history['avg_travel_time'][0],
            'final_avg_time': self.history['avg_travel_time'][-1],
            'time_improvement': (self.history['avg_travel_time'][0] -
                                self.history['avg_travel_time'][-1])
        }
