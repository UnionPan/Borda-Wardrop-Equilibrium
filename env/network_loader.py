"""
Network Loader Module

This module provides utilities for loading transportation network data from TNTP files.
Supports loading network topology (*_net.tntp), demand/trips (*_trips.tntp),
and node coordinates (*_node.tntp).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class NetworkLoader:
    """Loads transportation network data from TNTP format files."""

    def __init__(self, network_dir: str, network_name: Optional[str] = None):
        """
        Initialize network loader.

        Parameters
        ----------
        network_dir : str
            Path to the directory containing network files
        network_name : str, optional
            Base name of network files (e.g., 'SiouxFalls' for 'SiouxFalls_net.tntp')
            If None, will attempt to auto-detect from directory
        """
        self.network_dir = network_dir
        self.network_name = network_name or self._detect_network_name()

        # Data containers
        self.net_data = None
        self.trips_data = None
        self.node_data = None
        self.metadata = {}

    def _detect_network_name(self) -> str:
        """Auto-detect network name from available files."""
        files = os.listdir(self.network_dir)
        net_files = [f for f in files if f.endswith('_net.tntp')]

        if not net_files:
            raise ValueError(f"No *_net.tntp file found in {self.network_dir}")

        # Extract base name (remove _net.tntp suffix)
        base_name = net_files[0].replace('_net.tntp', '')
        return base_name

    def load_network(self) -> pd.DataFrame:
        """
        Load network topology from *_net.tntp file.

        Returns
        -------
        pd.DataFrame
            Network data with columns: init_node, term_node, capacity, length,
            free_flow_time, b, power, speed, toll, link_type
        """
        net_file = os.path.join(self.network_dir, f"{self.network_name}_net.tntp")

        if not os.path.exists(net_file):
            raise FileNotFoundError(f"Network file not found: {net_file}")

        with open(net_file, 'r') as f:
            lines = f.readlines()

        # Parse metadata
        metadata_end = 0
        for i, line in enumerate(lines):
            if '<END OF METADATA>' in line:
                metadata_end = i
                break
            if line.startswith('<'):
                # Parse metadata like <NUMBER OF ZONES> 24
                parts = line.strip().split('>')
                if len(parts) >= 2:
                    key = parts[0].replace('<', '').strip()
                    value = parts[1].strip()
                    self.metadata[key] = value

        # Find header line (starts with ~)
        header_line = None
        data_start = metadata_end + 1
        for i in range(metadata_end, len(lines)):
            if lines[i].strip().startswith('~') and 'init_node' in lines[i]:
                header_line = lines[i]
                data_start = i + 1
                break

        # Parse header
        if header_line:
            header = [col.strip() for col in header_line.strip().strip('~').split('\t')
                     if col.strip() and col.strip() != ';']
        else:
            # Default header if not found
            header = ['init_node', 'term_node', 'capacity', 'length', 'free_flow_time',
                     'b', 'power', 'speed', 'toll', 'link_type']

        # Parse data lines
        data_rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('~'):
                # Remove leading/trailing tabs and semicolons
                values = [val.strip() for val in line.split('\t')
                         if val.strip() and val.strip() != ';']
                if values:
                    data_rows.append(values)

        # Create DataFrame
        self.net_data = pd.DataFrame(data_rows, columns=header)

        # Convert numeric columns
        numeric_cols = ['capacity', 'length', 'free_flow_time', 'b', 'power', 'speed', 'toll', 'link_type']
        for col in numeric_cols:
            if col in self.net_data.columns:
                self.net_data[col] = pd.to_numeric(self.net_data[col], errors='coerce')

        # Convert node columns to integers
        for col in ['init_node', 'term_node']:
            if col in self.net_data.columns:
                self.net_data[col] = pd.to_numeric(self.net_data[col], errors='coerce').astype(int)

        return self.net_data

    def load_trips(self) -> np.ndarray:
        """
        Load OD demand matrix from *_trips.tntp file.

        Returns
        -------
        np.ndarray
            Origin-Destination demand matrix
        """
        trips_file = os.path.join(self.network_dir, f"{self.network_name}_trips.tntp")

        if not os.path.exists(trips_file):
            raise FileNotFoundError(f"Trips file not found: {trips_file}")

        with open(trips_file, 'r') as f:
            all_rows = f.read()

        # Split by 'Origin' keyword
        blocks = all_rows.split('Origin')[1:]

        # Parse OD matrix
        matrix = {}
        for block in blocks:
            lines = block.strip().split('\n')
            origin = int(lines[0].strip())

            # Parse destinations
            dest_lines = lines[1:]
            destinations = {}
            for line in dest_lines:
                # Parse format like: "1 : 0.0; 2 : 100.0; 3 : 100.0;"
                pairs = line.split(';')
                for pair in pairs:
                    if ':' in pair:
                        parts = pair.split(':')
                        dest = int(parts[0].strip())
                        flow = float(parts[1].strip())
                        destinations[dest] = flow

            matrix[origin] = destinations

        # Convert to numpy array
        if 'NUMBER OF ZONES' in self.metadata:
            num_zones = int(self.metadata['NUMBER OF ZONES'])
        else:
            num_zones = max(matrix.keys()) if matrix else 0
        
        if not num_zones:
            # Try to infer from destinations if metadata is missing
            max_dest = 0
            for origin in matrix:
                if matrix[origin]:
                    max_dest = max(max_dest, max(matrix[origin].keys()))
            num_zones = max(max(matrix.keys()) if matrix else 0, max_dest)

        od_matrix = np.zeros((num_zones, num_zones))

        for origin in matrix:
            for dest in matrix[origin]:
                # Convert to 0-indexed
                od_matrix[origin-1, dest-1] = matrix[origin][dest]

        self.trips_data = od_matrix
        return self.trips_data

    def load_nodes(self) -> Optional[pd.DataFrame]:
        """
        Load node coordinates from *_node.tntp file (if available).

        Returns
        -------
        pd.DataFrame or None
            Node data with columns: Node, X, Y
            Returns None if node file doesn't exist
        """
        node_file = os.path.join(self.network_dir, f"{self.network_name}_node.tntp")

        if not os.path.exists(node_file):
            print(f"Warning: Node file not found: {node_file}")
            return None

        # Read node file
        nodes = []
        with open(node_file, 'r') as f:
            lines = f.readlines()

        # Parse nodes (format: Node X Y ;)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Node'):
                parts = [p.strip() for p in line.split('\t') if p.strip() and p.strip() != ';']
                if len(parts) >= 3:
                    nodes.append(parts[:3])

        self.node_data = pd.DataFrame(nodes, columns=['Node', 'X', 'Y'])
        self.node_data['Node'] = self.node_data['Node'].astype(int)
        self.node_data['X'] = pd.to_numeric(self.node_data['X'])
        self.node_data['Y'] = pd.to_numeric(self.node_data['Y'])

        return self.node_data

    def load_all(self) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame]]:
        """
        Load all available network data.

        Returns
        -------
        tuple
            (network_data, od_matrix, node_coordinates)
        """
        net = self.load_network()
        trips = self.load_trips()
        nodes = self.load_nodes()

        return net, trips, nodes

    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Get list of edges as (init_node, term_node) tuples.

        Returns
        -------
        List[Tuple[int, int]]
            List of directed edges
        """
        if self.net_data is None:
            self.load_network()

        return list(zip(self.net_data['init_node'], self.net_data['term_node']))

    def get_free_flow_times(self) -> np.ndarray:
        """Get free flow travel times for all links."""
        if self.net_data is None:
            self.load_network()
        return self.net_data['free_flow_time'].values

    def get_capacities(self) -> np.ndarray:
        """Get capacities for all links."""
        if self.net_data is None:
            self.load_network()
        return self.net_data['capacity'].values

    def get_od_pairs(self) -> List[Tuple[str, str]]:
        """
        Get list of OD pairs with non-zero demand.

        Returns
        -------
        List[Tuple[str, str]]
            List of (origin, destination) pairs with positive demand
        """
        if self.trips_data is None:
            self.load_trips()

        od_pairs = []
        num_zones = self.trips_data.shape[0]
        for i in range(num_zones):
            for j in range(num_zones):
                if self.trips_data[i, j] > 0:
                    od_pairs.append((str(i+1), str(j+1)))  # Convert to string

        return od_pairs

    def get_od_demands(self, od_pairs: Optional[List[Tuple[str, str]]] = None) -> np.ndarray:
        """
        Get demands for specified OD pairs.

        Parameters
        ----------
        od_pairs : List[Tuple[int, int]], optional
            List of (origin, destination) pairs. If None, returns all non-zero demands.

        Returns
        -------
        np.ndarray
            Array of demands corresponding to OD pairs
        """
        if self.trips_data is None:
            self.load_trips()

        if od_pairs is None:
            od_pairs = self.get_od_pairs()

        demands = []
        for origin, dest in od_pairs:
            # Convert to 0-indexed
            demands.append(self.trips_data[int(origin)-1, int(dest)-1])

        return np.array(demands)


def discover_networks(networks_dir: str = 'networks') -> Dict[str, str]:
    """
    Discover all available networks in the networks directory.

    Parameters
    ----------
    networks_dir : str
        Path to networks directory

    Returns
    -------
    Dict[str, str]
        Dictionary mapping network names to their directory paths
    """
    networks = {}

    if not os.path.exists(networks_dir):
        return networks

    for item in os.listdir(networks_dir):
        item_path = os.path.join(networks_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains *_net.tntp file
            files = os.listdir(item_path)
            net_files = [f for f in files if f.endswith('_net.tntp')]
            if net_files:
                networks[item] = item_path

    return networks
