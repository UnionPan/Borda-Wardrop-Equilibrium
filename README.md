# Borda-Wardrop-Equilibrium

The simulation for attacking Wardrop equilibrium by redistributing the traffic flow information acquired by the central planner.

![expweight_365day_flow_animation](./results/expweight_365day_flow_animation.gif)

## Features

*   **Modular Architecture:** A clean, modular structure that separates concerns for the environment, algorithms, attacks, and utilities.
*   **Traffic Network Environments:** Supports both non-atomic (continuous flow) and atomic (discrete agent) congestion games.
*   **Extensive Network Support:** Comes with over 20 transportation networks in TNTP format, including Sioux Falls, Chicago, and Berlin.
*   **Flexible Algorithms:** Implements classic algorithms like Frank-Wolfe and Exponential Weights for traffic assignment.
*   **Advanced Visualization:** Includes enhanced geographic visualization capabilities with OpenStreetMap backgrounds, flow-based edge coloring, and automatic coordinate detection.

## Directory Structure

```
Borda-Wardrop-Equilibrium/
├── env/                    # Environment & Traffic Network Models
│   ├── graph.py           # Graph and TrafficNetwork classes
│   ├── network_loader.py  # TNTP file loader
│   ├── traffic_model.py   # TrafficNetworkEnvironment with visualization
│   └── __init__.py
│
├── algorithms/             # Algorithm Implementations  
│   ├── frank_wolfe.py     # Frank-Wolfe algorithm
│   ├── system_optimum.py  # System Optimum algorithm
│   └── exp_weight.py      # EXP3 algorithm
│
├── attacks/                # Attack Strategies
│   └── wardrop_attack.py  # Wardrop equilibrium attack
│
├── utils/                  # Utility Functions
│   └── visualization.py   # Visualization utilities
│
├── scripts/                # Shell Scripts for Experiments
│   └── run_attack_experiment.sh
│
├── tests/                  # Test Suite
│   └── test_network_loading.py
│
└── networks/               # 23 Traffic Network Datasets
    ├── SiouxFalls/
    ├── Anaheim/
    └── ... (and 20+ more)
```

## Supported Networks

The system currently supports over 20 transportation networks, including:

*   **North American Networks:** Sioux Falls, Chicago (Sketch & Regional), Anaheim, Austin, Philadelphia, Winnipeg, Eastern Massachusetts
*   **European Networks:** Berlin (5 regions), Barcelona, Birmingham (England), Hessen, Terrassa
*   **Asia-Pacific:** Sydney, Gold Coast
*   **Test Cases:** Braess Example, SymmetricaTestCase

## Getting Started

### Quick Start

```python
# 1. Discover available networks
from env.network_loader import discover_networks
networks = discover_networks()

# 2. Load a network
from env.traffic_model import NonAtomicTrafficEnvironment
env = NonAtomicTrafficEnvironment('networks/SiouxFalls')

# 3. View network info
env.print_network_info()

# 4. Visualize
env.visualize()

# 5. Get data for algorithms
graph, origins, dests, demands, times, caps = env.get_graph_data()
```

### Running an Experiment

An example script `run_siouxfalls_experiment.py` is provided to demonstrate how to run the implemented algorithms on a network. To run the experiment:

```bash
python3 run_siouxfalls_experiment.py
```

This will run the Frank-Wolfe, Exponential Weights, and System Optimum algorithms on the specified network and generate convergence plots.

## Traffic Network Environments

The project provides two traffic assignment environments to support different congestion game models:

1.  **Non-Atomic Congestion Game:** Continuous flow model for classic traffic assignment problems.
2.  **Atomic Congestion Game:** Discrete agents model for strategic routing games.

Both environments are available in `env/traffic_model.py`.

### Non-Atomic Traffic Environment

*   **Concept:** Traffic is modeled as infinitesimal, continuous flow.
*   **Equilibrium:** Wardrop Equilibrium (User Equilibrium / System Optimum).
*   **Usage:** Ideal for large-scale population routing and analysis of algorithms like Frank-Wolfe and Exponential Weights.

### Atomic Traffic Environment

*   **Concept:** Traffic consists of discrete agents whose decisions measurably affect congestion.
*   **Equilibrium:** Nash Equilibrium.
*   **Usage:** Suited for strategic routing games, best-response dynamics, and game-theoretic analysis.

## Simulation Interface

The `NonAtomicTrafficEnvironment` class provides a complete simulation interface for traffic assignment algorithms. It receives traffic distributions, simulates travel times using BPR performance functions, and returns performance feedback.

### Key Methods

*   `step(path_flow)`: Simulates one step with a given path flow distribution.
*   `get_shortest_path_flow(link_time)`: Performs an all-or-nothing assignment to the shortest paths.
*   `compute_beckmann_objective(link_flow)`: Computes the Beckmann objective for User Equilibrium.
*   `compute_system_cost(link_flow)`: Computes the total system travel time for System Optimum.

## Enhanced Geographic Visualization

The project includes enhanced geographic visualization capabilities that display traffic networks on actual maps with OpenStreetMap backgrounds.

### Features

*   **Geographic Coordinates:** Automatic detection of lat/lon coordinates.
*   **Map Backgrounds:** OpenStreetMap, Stamen, and CartoDB tiles.
*   **Flow Visualization:** Color-coded edges based on traffic flow.
*   **Node Coloring:** Origins (red), Destinations (blue), and Transfer nodes (green).

### Usage

```python
from env.traffic_model import NonAtomicTrafficEnvironment

# Load a network with coordinates
env = NonAtomicTrafficEnvironment('networks/SiouxFalls')

# Visualize the network on a map
env.visualize(
    show_background=True,
    show_node_labels=True,
    save_path='sioux_falls_map.png'
)
```

## Running Tests

To run the network loading tests:

```bash
python tests/test_network_loading.py
```

## Development Status

This project has been restructured into a modular architecture. The core components, including the network loader, graph data structures, and traffic network environments, are complete and tested. The next phase of development will focus on migrating the remaining legacy code and implementing additional algorithms and attack strategies.
