"""
Base Traffic Network Environment

Provides common functionality for both atomic and non-atomic traffic models.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from collections import OrderedDict

from env.network_loader import NetworkLoader
from env.graph import TrafficNetwork


class BaseTrafficEnvironment:
    """
    Base class for traffic network environments.
    
    Provides:
    - Network loading and topology
    - BPR performance functions
    - Visualization
    - Common utilities
    """
    
    def __init__(self,
                 network_path: str,
                 network_name: Optional[str] = None,
                 alpha: float = 0.15,
                 beta: float = 4.0,
                 max_paths_per_od: Optional[int] = 10):
        """
        Initialize base traffic environment.

        Parameters
        ----------
        network_path : str
            Path to directory containing network files
        network_name : str, optional
            Base name of network files (auto-detected if not provided)
        alpha : float, default=0.15
            BPR function parameter (congestion sensitivity)
        beta : float, default=4.0
            BPR function parameter (congestion exponent)
        max_paths_per_od : int, optional, default=10
            Maximum paths to enumerate per OD pair. Set to None to find ALL paths
            (WARNING: can be very slow for dense networks!)
            Recommended: 10-15 for good performance, 20-30 for higher accuracy
        """
        self.network_path = network_path
        self.loader = NetworkLoader(network_path, network_name)
        # Expose network identifier for downstream reporting and plotting
        self.network_name = self.loader.network_name
        self.__name__ = self.network_name

        # BPR performance function parameters
        self.alpha = alpha
        self.beta = beta

        # Load network data
        self.net_data, self.od_matrix, self.node_coords = self.loader.load_all()

        # Extract network attributes as numpy arrays
        self.free_flow_times = self.loader.get_free_flow_times()
        self.capacities = self.loader.get_capacities()
        self.num_links = len(self.free_flow_times)

        # Create graph representation
        self.graph_dict = self._build_graph_dict()

        # Extract origins and destinations with non-zero demand
        self.origins, self.destinations = self._extract_od_nodes()

        # Extract OD demands
        self.od_pairs = self.loader.get_od_pairs()
        self.demands = self.loader.get_od_demands(self.od_pairs)
        self.num_od_pairs = len(self.od_pairs)

        # Create TrafficNetwork instance (for path enumeration and LP matrix)
        # PERFORMANCE: Limit paths to prevent exponential explosion
        print(f"Initializing TrafficNetwork (max_paths_per_od={max_paths_per_od})...")
        self.traffic_network = TrafficNetwork(
            graph=self.graph_dict,
            O=self.origins,
            D=self.destinations,
            max_paths_per_od=max_paths_per_od
        )
        
        # Link-Path incidence matrix
        self.LP_matrix = self.traffic_network.LP_matrix()
        self.num_paths = self.traffic_network.num_of_paths()
        
        # NetworkX graph for advanced operations and visualization
        self.nx_graph = self._build_networkx_graph()
        
        # Current state (for tracking)
        self.current_link_flow = None
        self.current_link_time = None
        
    def _build_graph_dict(self) -> OrderedDict:
        """Build ordered dictionary representation of graph."""
        graph_dict = OrderedDict()
        
        for _, row in self.net_data.iterrows():
            init_node = str(int(row['init_node']))
            term_node = str(int(row['term_node']))
            
            if init_node not in graph_dict:
                graph_dict[init_node] = []
            graph_dict[init_node].append(term_node)
            
            if term_node not in graph_dict:
                graph_dict[term_node] = []
                
        return graph_dict
    
    def _extract_od_nodes(self) -> Tuple[List[str], List[str]]:
        """Extract origin and destination nodes from OD matrix."""
        origins = set()
        destinations = set()
        
        num_zones = self.od_matrix.shape[0]
        for i in range(num_zones):
            for j in range(num_zones):
                if self.od_matrix[i, j] > 0:
                    origins.add(str(i + 1))
                    destinations.add(str(j + 1))
                    
        return sorted(list(origins)), sorted(list(destinations))
    
    def _build_networkx_graph(self) -> nx.MultiDiGraph:
        """Build NetworkX graph for visualization."""
        G = nx.MultiDiGraph()
        
        for node in self.graph_dict.keys():
            G.add_node(node)
            
        for idx, row in self.net_data.iterrows():
            init_node = str(int(row['init_node']))
            term_node = str(int(row['term_node']))
            
            G.add_edge(
                init_node,
                term_node,
                capacity=row['capacity'],
                length=row['length'],
                free_flow_time=row['free_flow_time'],
                b=row['b'],
                power=row['power'],
                edge_id=idx
            )
        
        if self.node_coords is not None:
            pos_dict = {}
            for _, row in self.node_coords.iterrows():
                node = str(int(row['Node']))
                pos_dict[node] = (row['X'], row['Y'])
            nx.set_node_attributes(G, pos_dict, 'pos')
            
        for node in G.nodes():
            if node in self.origins and node in self.destinations:
                G.nodes[node]['node_type'] = 'both'
                G.nodes[node]['color'] = 'purple'
            elif node in self.origins:
                G.nodes[node]['node_type'] = 'origin'
                G.nodes[node]['color'] = 'red'
            elif node in self.destinations:
                G.nodes[node]['node_type'] = 'destination'
                G.nodes[node]['color'] = 'blue'
            else:
                G.nodes[node]['node_type'] = 'transfer'
                G.nodes[node]['color'] = 'green'
                
        return G
    
    def compute_link_travel_time(self, link_flow: np.ndarray) -> np.ndarray:
        """
        Compute link travel times using BPR performance function.
        
        BPR function: t(f) = t0 * (1 + alpha * (f / capacity)^beta)
        
        Parameters
        ----------
        link_flow : np.ndarray, shape (num_links,)
            Flow on each link
            
        Returns
        -------
        np.ndarray, shape (num_links,)
            Travel time on each link
        """
        link_time = self.free_flow_times * (
            1.0 + self.alpha * np.power(link_flow / self.capacities, self.beta)
        )
        
        self.current_link_flow = link_flow.copy()
        self.current_link_time = link_time.copy()
        
        return link_time
    
    def compute_path_travel_time(self, link_time: np.ndarray) -> np.ndarray:
        """
        Compute path travel times from link travel times.
        
        path_time = LP_matrix^T @ link_time
        
        Parameters
        ----------
        link_time : np.ndarray, shape (num_links,)
            Travel time on each link
            
        Returns
        -------
        np.ndarray, shape (num_paths,)
            Travel time on each path
        """
        path_time = link_time.dot(self.LP_matrix)
        return path_time
    
    def convert_path_flow_to_link_flow(self, path_flow: np.ndarray) -> np.ndarray:
        """
        Convert path flows to link flows.
        
        link_flow = LP_matrix @ path_flow
        
        Parameters
        ----------
        path_flow : np.ndarray, shape (num_paths,)
            Flow on each path
            
        Returns
        -------
        np.ndarray, shape (num_links,)
            Flow on each link
        """
        link_flow = self.LP_matrix.dot(path_flow)
        return link_flow
    
    def get_network_stats(self) -> Dict:
        """Get basic statistics about the network."""
        stats = {
            'num_nodes': self.nx_graph.number_of_nodes(),
            'num_edges': self.nx_graph.number_of_edges(),
            'num_origins': len(self.origins),
            'num_destinations': len(self.destinations),
            'num_od_pairs': len(self.traffic_network.OD_pairs()),
            'num_paths': self.traffic_network.num_of_paths(),
            'total_demand': self.od_matrix.sum(),
            'avg_degree': sum(dict(self.nx_graph.degree()).values()) / self.nx_graph.number_of_nodes(),
        }
        
        if 'NUMBER OF ZONES' in self.loader.metadata:
            stats['num_zones'] = self.loader.metadata['NUMBER OF ZONES']
            
        return stats
    
    def print_network_info(self) -> None:
        """Print comprehensive network information."""
        print(f"\n{'='*60}")
        print(f"Traffic Network: {self.loader.network_name}")
        print(f"{'='*60}")
        
        stats = self.get_network_stats()
        print(f"\nNetwork Statistics:")
        print(f"  Nodes:            {stats['num_nodes']}")
        print(f"  Edges:            {stats['num_edges']}")
        print(f"  Origins:          {stats['num_origins']}")
        print(f"  Destinations:     {stats['num_destinations']}")
        print(f"  OD Pairs:         {stats['num_od_pairs']}")
        print(f"  Paths:            {stats['num_paths']}")
        print(f"  Total Demand:     {stats['total_demand']:.0f}")
        
        print(f"\nNetwork Properties:")
        print(f"  Has Coordinates:  {'Yes' if self.node_coords is not None else 'No'}")
        print(f"  Avg Capacity:     {self.net_data['capacity'].mean():.2f}")
        print(f"  Avg Free Time:    {self.net_data['free_flow_time'].mean():.2f}")
        print(f"  BPR Parameters:   α={self.alpha}, β={self.beta}")
        
        print(f"{'='*60}\n")
    
    def visualize(self,
                  figsize: Tuple[int, int] = (14, 10),
                  node_size: int = 500,
                  edge_width: float = 2.0,
                  show_background: bool = True,
                  show_node_labels: bool = False,
                  flow_data: Optional[np.ndarray] = None,
                  title: Optional[str] = None,
                  save_path: Optional[str] = None,
                  use_simple: bool = False) -> None:
        """
        Visualize the traffic network.

        If use_simple=False and geospatial libraries available,
        uses enhanced geographic visualization with map background.
        Otherwise falls back to simple NetworkX visualization.

        Parameters
        ----------
        figsize : tuple
            Figure size
        node_size : int
            Size of nodes
        edge_width : float
            Width of edges
        show_background : bool
            Show geographic map background (if coordinates available)
        show_node_labels : bool
            Show node ID labels
        flow_data : np.ndarray, optional
            Link flows for visualization
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
        use_simple : bool
            Force simple visualization (no geographic features)
        """
        if not use_simple:
            try:
                from utils.visualization import NetworkVisualizer
                visualizer = NetworkVisualizer(self)
                visualizer.plot_network_on_map(
                    figsize=figsize,
                    node_size=node_size,
                    edge_width=edge_width,
                    show_background=show_background,
                    show_node_labels=show_node_labels,
                    flow_data=flow_data,
                    title=title,
                    save_path=save_path
                )
                return
            except Exception as e:
                print(f"Enhanced visualization failed: {e}")
                print("Falling back to simple visualization...")

        # Simple fallback visualization
        fig, ax = plt.subplots(figsize=figsize)

        if self.node_coords is not None:
            pos = nx.get_node_attributes(self.nx_graph, 'pos')
        else:
            pos = nx.spring_layout(self.nx_graph, seed=42)

        node_colors = [self.nx_graph.nodes[node].get('color', 'gray')
                      for node in self.nx_graph.nodes()]

        nx.draw_networkx_nodes(self.nx_graph, pos, node_color=node_colors,
                              node_size=node_size, ax=ax)
        if show_node_labels:
            nx.draw_networkx_labels(self.nx_graph, pos, font_size=10,
                                   font_color='white', font_weight='bold', ax=ax)
        nx.draw_networkx_edges(self.nx_graph, pos, width=edge_width, alpha=0.6,
                              arrows=True, arrowsize=15, arrowstyle='->', ax=ax)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Origin'),
            Patch(facecolor='blue', label='Destination'),
            Patch(facecolor='purple', label='Origin & Destination'),
            Patch(facecolor='green', label='Transfer Node')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        if title is None:
            title = f"Traffic Network: {self.loader.network_name}"
        ax.set_title(title, fontsize=16, fontweight='bold')

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()
