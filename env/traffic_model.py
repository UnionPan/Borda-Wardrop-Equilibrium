"""
Traffic Network Environment Models

Provides two types of traffic assignment environments:
1. Non-Atomic: Continuous flow, probability distributions over paths
2. Atomic: Homogeneous populations with optional background traffic
"""

import numpy as np
from typing import List, Optional, Tuple
from env.traffic_base import BaseTrafficEnvironment


class NonAtomicTrafficEnvironment(BaseTrafficEnvironment):
    """
    Non-Atomic Congestion Game Environment.
    
    Traffic is modeled as **infinitesimal** (continuous flow).
    - Each OD pair has a demand volume
    - Flow is distributed across paths according to probability distributions
    - Classic Wardrop equilibrium setup
    - Used for Frank-Wolfe, Exp-Weight algorithms
    
    Interface:
    - Algorithms provide: path flow distributions (continuous)
    - Environment returns: link flows, link times, path times
    """
    
    def __init__(self, network_path: str, network_name: Optional[str] = None,
                 alpha: float = 0.15, beta: float = 4.0, max_paths_per_od: Optional[int] = 10):
        """
        Initialize non-atomic traffic environment.

        Parameters
        ----------
        network_path : str
            Path to network directory
        network_name : str, optional
            Network name (auto-detected if None)
        alpha, beta : float
            BPR parameters
        max_paths_per_od : int, optional
            Max paths to enumerate per OD pair (default: 10)
        """
        super().__init__(network_path, network_name, alpha, beta, max_paths_per_od)
        
        # State for non-atomic model
        self.current_path_flow = None
        self.current_path_time = None
        
    def step(self, path_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate one step with path flow distribution.
        
        Parameters
        ----------
        path_flow : np.ndarray, shape (num_paths,)
            Flow on each path (continuous distribution)
            
        Returns
        -------
        link_flow : np.ndarray, shape (num_links,)
            Resulting flow on each link
        link_time : np.ndarray, shape (num_links,)
            Travel time on each link (BPR)
        path_time : np.ndarray, shape (num_paths,)
            Travel time on each path
        """
        # Convert path flows to link flows
        link_flow = self.convert_path_flow_to_link_flow(path_flow)
        
        # Compute link travel times (BPR)
        link_time = self.compute_link_travel_time(link_flow)
        
        # Compute path travel times
        path_time = self.compute_path_travel_time(link_time)
        
        # Update state
        self.current_path_flow = path_flow.copy()
        self.current_path_time = path_time.copy()
        
        return link_flow, link_time, path_time
    
    def step_from_link_flow(self, link_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate from link flows (for algorithms that work in link space).
        
        Parameters
        ----------
        link_flow : np.ndarray, shape (num_links,)
            Flow on each link
            
        Returns
        -------
        link_time : np.ndarray, shape (num_links,)
            Travel time on each link
        path_time : np.ndarray, shape (num_paths,)
            Travel time on each path
        """
        link_time = self.compute_link_travel_time(link_flow)
        path_time = self.compute_path_travel_time(link_time)
        
        return link_time, path_time
    
    def get_shortest_path_flow(self, link_time: np.ndarray) -> np.ndarray:
        """
        All-or-nothing assignment: assign all demand to shortest paths.
        
        For each OD pair, finds the shortest path and assigns all demand.
        Used in Frank-Wolfe direction finding.
        
        Parameters
        ----------
        link_time : np.ndarray, shape (num_links,)
            Current link travel times
            
        Returns
        -------
        np.ndarray, shape (num_paths,)
            Path flow (all demand on shortest paths)
        """
        path_time = self.compute_path_travel_time(link_time)
        path_flow = np.zeros(self.num_paths)
        
        paths_category = self.traffic_network.paths_category()
        
        for od_idx in range(self.num_od_pairs):
            # Find paths for this OD pair
            path_indices = [i for i, cat in enumerate(paths_category) if cat == od_idx]
            
            # Find shortest path
            od_path_times = path_time[path_indices]
            min_idx = np.argmin(od_path_times)
            shortest_path_idx = path_indices[min_idx]
            
            # Assign all demand to shortest path
            path_flow[shortest_path_idx] = self.demands[od_idx]
        
        return path_flow
    
    def compute_beckmann_objective(self, link_flow: np.ndarray) -> float:
        """
        Compute Beckmann objective (User Equilibrium).
        
        Φ(f) = Σ_a ∫_0^{f_a} t_a(x) dx
        
        For BPR: ∫ t0*(1 + α*(x/c)^β) dx 
                = t0*f + α*t0*f/(β+1) * (f/c)^β
        """
        term1 = self.free_flow_times * link_flow
        term2 = (self.alpha * self.free_flow_times * link_flow / (self.beta + 1)) * \
                np.power(link_flow / self.capacities, self.beta)
        return np.sum(term1 + term2)
    
    def compute_system_cost(self, link_flow: np.ndarray) -> float:
        """
        Compute total system travel time (System Optimum).
        
        Total cost = Σ_a f_a * t_a(f_a)
        """
        link_time = self.compute_link_travel_time(link_flow)
        return np.sum(link_flow * link_time)
    
    def get_average_travel_time(self) -> float:
        """Get average travel time per unit flow."""
        if self.current_link_flow is None or self.current_link_time is None:
            raise ValueError("No flow assignment made. Call step() first.")
        
        total_system_time = np.sum(self.current_link_flow * self.current_link_time)
        total_demand = np.sum(self.demands)
        
        return total_system_time / total_demand



class AtomicTrafficEnvironment(NonAtomicTrafficEnvironment):
    """
    Population-level congestion game environment with background traffic.

    The original OD demand is split into:
      - background demand: fixed flow pattern (legacy users)
      - strategic demand: homogeneous population controlled by algorithms
    """

    def __init__(self,
                 network_path: str,
                 network_name: Optional[str] = None,
                 background_ratio: float = 0.8,
                 alpha: float = 0.15,
                 beta: float = 4.0,
                 max_paths_per_od: Optional[int] = 10):
        """
        Initialize population-based atomic environment.

        Parameters
        ----------
        network_path : str
            Path to network directory
        network_name : str, optional
            Network name (auto-detected if None)
        background_ratio : float, default=0.8
            Fraction of OD demand treated as fixed background flow
        alpha, beta : float
            BPR parameters
        """
        super().__init__(network_path, network_name, alpha, beta, max_paths_per_od)

        if not (0.0 <= background_ratio < 1.0):
            raise ValueError("background_ratio must lie in [0, 1).")

        self.background_ratio = background_ratio

        # Preserve original demand data
        self.total_demand = float(np.sum(self.demands))
        self.original_demands = self.demands.astype(float)

        # Split background vs strategic demand
        self.background_demand = self.original_demands * self.background_ratio
        self.strategic_demand = self.original_demands - self.background_demand

        # Expose strategic demand as operative demand for algorithms
        self.demands = self.strategic_demand.copy()
        self.strategic_total_demand = float(np.sum(self.strategic_demand))

        # Cached background flow (path space)
        self.background_path_flow: Optional[np.ndarray] = None

        # Cached path indices per OD pair for fast lookups
        paths_category = self.traffic_network.paths_category()
        self.od_path_indices: List[np.ndarray] = [
            np.array([i for i, cat in enumerate(paths_category) if cat == od_idx], dtype=int)
            for od_idx in range(self.num_od_pairs)
        ]

        # Track strategic flow state
        self.current_strategic_path_flow: Optional[np.ndarray] = None
        self.current_total_path_flow: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Background flow utilities
    # ------------------------------------------------------------------
    def initialize_background_uniformly(self):
        """Distribute background demand uniformly across available paths."""
        background_path_flow = np.zeros(self.num_paths)

        for od_idx, path_indices in enumerate(self.od_path_indices):
            if path_indices.size == 0:
                continue
            demand = float(self.background_demand[od_idx])
            if demand <= 0:
                continue
            background_path_flow[path_indices] = demand / path_indices.size

        self.background_path_flow = background_path_flow

    def set_background_flow(self, background_path_flow: np.ndarray):
        """Set explicit background path flow (must match background demand)."""
        background_path_flow = np.asarray(background_path_flow, dtype=float)
        if background_path_flow.shape[0] != self.num_paths:
            raise ValueError("Background flow must have length equal to num_paths.")

        background_total = np.sum(background_path_flow)
        expected_total = float(np.sum(self.background_demand))
        if expected_total > 0 and not np.isclose(background_total, expected_total, atol=1e-6):
            raise ValueError(
                f"Background flow totals {background_total:.6f}, expected {expected_total:.6f}."
            )

        self.background_path_flow = background_path_flow

    def get_background_flow(self) -> np.ndarray:
        """Return the current background path flow, initializing if necessary."""
        if self.background_path_flow is None:
            self.initialize_background_uniformly()
        return self.background_path_flow.copy()

    # ------------------------------------------------------------------
    # Strategic flow helpers
    # ------------------------------------------------------------------
    def zero_strategic_flow(self) -> np.ndarray:
        """Return zero strategic flow vector."""
        return np.zeros(self.num_paths, dtype=float)

    def uniform_strategic_flow(self) -> np.ndarray:
        """Return strategic flow distributed uniformly across paths per OD pair."""
        strategic_flow = np.zeros(self.num_paths, dtype=float)
        for od_idx, path_indices in enumerate(self.od_path_indices):
            demand = float(self.strategic_demand[od_idx])
            if demand <= 0 or path_indices.size == 0:
                continue
            strategic_flow[path_indices] = demand / path_indices.size
        return strategic_flow

    def flow_from_distribution(self, path_probabilities: np.ndarray) -> np.ndarray:
        """
        Convert per-path probabilities into strategic flow.

        Parameters
        ----------
        path_probabilities : np.ndarray
            Probabilities for each path. For every OD pair, the probabilities
            on its paths must sum to 1 (or 0 if the strategic demand is zero).
        """
        path_probabilities = np.asarray(path_probabilities, dtype=float)
        if path_probabilities.shape[0] != self.num_paths:
            raise ValueError("Probability vector must match number of paths.")

        strategic_flow = np.zeros(self.num_paths, dtype=float)
        for od_idx, path_indices in enumerate(self.od_path_indices):
            demand = float(self.strategic_demand[od_idx])
            if demand <= 0 or path_indices.size == 0:
                continue
            probs = path_probabilities[path_indices]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            strategic_flow[path_indices] = demand * probs

        return strategic_flow

    def distribution_from_flow(self, strategic_path_flow: np.ndarray) -> np.ndarray:
        """Convert a strategic flow vector into per-path probabilities."""
        strategic_path_flow = np.asarray(strategic_path_flow, dtype=float)
        if strategic_path_flow.shape[0] != self.num_paths:
            raise ValueError("Strategic path flow must have length equal to num_paths.")

        probabilities = np.zeros_like(strategic_path_flow)
        for od_idx, path_indices in enumerate(self.od_path_indices):
            demand = float(self.strategic_demand[od_idx])
            if demand <= 0 or path_indices.size == 0:
                continue
            flow_segment = strategic_path_flow[path_indices]
            flow_total = flow_segment.sum()
            if flow_total > 0:
                probabilities[path_indices] = flow_segment / flow_total

        return probabilities

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------
    def step(self, strategic_path_flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the network given strategic path flows.

        Parameters
        ----------
        strategic_path_flow : np.ndarray
            Path flows contributed by the strategic population only.
        """
        strategic_path_flow = np.asarray(strategic_path_flow, dtype=float)
        if strategic_path_flow.shape[0] != self.num_paths:
            raise ValueError("Strategic path flow must have length equal to num_paths.")
        if np.any(strategic_path_flow < -1e-12):
            raise ValueError("Strategic path flow must be non-negative.")

        if self.background_path_flow is None:
            self.initialize_background_uniformly()

        total_path_flow = self.background_path_flow + strategic_path_flow

        link_flow = self.convert_path_flow_to_link_flow(total_path_flow)
        link_time = self.compute_link_travel_time(link_flow)
        path_time = self.compute_path_travel_time(link_time)

        self.current_path_flow = total_path_flow.copy()
        self.current_link_flow = link_flow.copy()
        self.current_path_time = path_time.copy()
        self.current_link_time = link_time.copy()
        self.current_strategic_path_flow = strategic_path_flow.copy()
        self.current_total_path_flow = total_path_flow

        return link_flow, link_time, path_time

    def get_current_strategic_flow(self) -> np.ndarray:
        """Return the most recent strategic path flow vector."""
        if self.current_strategic_path_flow is None:
            raise ValueError("No strategic flow evaluated yet. Call step() first.")
        return self.current_strategic_path_flow.copy()

    def get_total_path_flow(self) -> np.ndarray:
        """Return total (background + strategic) path flow."""
        if self.current_total_path_flow is None:
            raise ValueError("No flow evaluated yet. Call step() first.")
        return self.current_total_path_flow.copy()

    def get_average_travel_time(self) -> float:
        """
        Override to report average travel time per unit of TOTAL demand
        (background + strategic).
        """
        if self.current_link_flow is None or self.current_link_time is None:
            raise ValueError("No flow assignment made. Call step() first.")

        total_system_time = np.sum(self.current_link_flow * self.current_link_time)
        if self.total_demand <= 0:
            return 0.0
        return total_system_time / self.total_demand
