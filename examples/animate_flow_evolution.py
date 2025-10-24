"""
Visualize Flow Evolution Animation on OSM Map

Creates animated GIFs showing how traffic flows evolve during optimization.
The "temperature" (congestion level) of each link is shown as colors on the map.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from env.traffic_model import NonAtomicTrafficEnvironment
from algorithms.exp_weight import ExpWeight
from algorithms.frank_wolfe import FrankWolfe
from algorithms.system_optimum import SystemOptimum
from utils.visualization import animate_flow_evolution


def run_animation_demo():
    """Run algorithms and create flow evolution animations."""

    print("="*70)
    print("Flow Evolution Animation Demo")
    print("="*70)
    print()

    # Load Sioux Falls network
    print("Loading Sioux Falls network...")
    network_path = "./networks/SiouxFalls"
    env = NonAtomicTrafficEnvironment(network_path=network_path)

    print(f"Network loaded:")
    print(f"  Nodes: {env.nx_graph.number_of_nodes()}")
    print(f"  Links: {env.num_links}")
    print(f"  Paths: {env.num_paths}")
    print(f"  OD Pairs: {env.num_od_pairs}")
    print(f"  Total Demand: {env.demands.sum():.0f}")
    print()

    # ===================================================================
    # ExpWeight Animation (365 frames for "365 days")
    # ===================================================================
    print("="*70)
    print("1. ExpWeight Algorithm Animation")
    print("="*70)

    ew = ExpWeight(
        env,
        max_iterations=200,  # 365 iterations for "365 days"
        learning_rate=5.63,
        tolerance=1e-10
    )

    # IMPORTANT: Enable full array storage for animation
    ew.store_full_arrays = True

    print("Running ExpWeight (365 iterations)...")
    print("This will take a few minutes...")
    results_ew = ew.run(verbose=False)

    print(f"\nExpWeight Results:")
    print(f"  Iterations: {results_ew['iterations']}")
    print(f"  Initial Beckmann: {ew.history['beckmann_potential'][0]:.2e}")
    print(f"  Final Beckmann: {results_ew['final_beckmann']:.2e}")
    print(f"  Improvement: {ew.history['beckmann_potential'][0] / results_ew['final_beckmann']:.2f}x")
    print()

    # Create animation (sample every 5 frames for faster rendering)
    print("Creating ExpWeight flow animation with convergence plot...")
    print("(This will generate ~73 frames, may take several minutes)")
    animate_flow_evolution(
        env,
        ew,
        output_path='./results/expweight_365day_flow_animation.gif',
        fps=5,  # 10 frames per second
        interval_frames=1,  # Show every 5th iteration
        figsize=(20, 10),  # Wide format for side-by-side layout
        show_convergence_plot=True
    )
    print()

    # ===================================================================
    # Frank-Wolfe Animation (shorter, 100 iterations)
    # ===================================================================
    print("="*70)
    print("2. Frank-Wolfe Algorithm Animation")
    print("="*70)

    fw = FrankWolfe(
        env,
        max_iterations=200,
        tolerance=1e-10,
        step_size_method='optimal'
    )

    fw.store_full_arrays = True

    print("Running Frank-Wolfe (100 iterations)...")
    results_fw = fw.run(verbose=False)

    print(f"\nFrank-Wolfe Results:")
    print(f"  Iterations: {results_fw['iterations']}")
    print(f"  Initial Beckmann: {fw.history['beckmann_potential'][0]:.2e}")
    print(f"  Final Beckmann: {results_fw['final_beckmann']:.2e}")
    print(f"  Improvement: {fw.history['beckmann_potential'][0] / results_fw['final_beckmann']:.2f}x")
    print()

    print("Creating Frank-Wolfe flow animation with convergence plot...")
    print("(This will generate ~20 frames)")
    animate_flow_evolution(
        env,
        fw,
        output_path='./results/frankwolfe_flow_animation.gif',
        fps=5,
        interval_frames=1,
        figsize=(20, 10),
        show_convergence_plot=True
    )
    print()

    # ===================================================================
    # System Optimum Animation (100 iterations)
    # ===================================================================
    print("="*70)
    print("3. System Optimum Algorithm Animation")
    print("="*70)

    so = SystemOptimum(
        env,
        max_iterations=200,
        tolerance=1e-10
    )

    so.store_full_arrays = True

    print("Running System Optimum (365 iterations)...")
    results_so = so.run(verbose=False)

    print(f"\nSystem Optimum Results:")
    print(f"  Iterations: {results_so['iterations']}")
    print(f"  Initial System Cost: {so.history['system_cost'][0]:.2e}")
    print(f"  Final System Cost: {results_so['final_system_cost']:.2e}")
    print(f"  Improvement: {so.history['system_cost'][0] / results_so['final_system_cost']:.2f}x")
    print()

    print("Creating System Optimum flow animation with convergence plot...")
    print("(This will generate ~20 frames)")
    animate_flow_evolution(
        env,
        so,
        output_path='./results/system_optimum_flow_animation.gif',
        fps=5,
        interval_frames=1,
        figsize=(20, 10),
        show_convergence_plot=True
    )
    print()

    # ===================================================================
    # Summary
    # ===================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Three animated GIFs have been created in ./results/:")
    print()
    print("1. expweight_365day_flow_animation.gif")
    print("   - Shows 365 iterations of ExpWeight algorithm")
    print("   - ~73 frames (every 5th iteration)")
    print("   - 10 fps")
    print()
    print("2. frankwolfe_flow_animation.gif")
    print("   - Shows 365 iterations of Frank-Wolfe algorithm")
    print("   - ~20 frames")
    print("   - 5 fps")
    print()
    print("3. system_optimum_flow_animation.gif")
    print("   - Shows 365 iterations of System Optimum")
    print("   - ~20 frames")
    print("   - 5 fps")
    print()
    print("Each animation shows TWO PANELS:")
    print()
    print("LEFT PANEL - Network Visualization:")
    print("  - High-contrast flow colors:")
    print("    * White/Yellow = light traffic")
    print("    * Orange/Red = moderate congestion")
    print("    * Dark Red/Black = severe congestion")
    print("  - Line thickness = congestion level")
    print("  - Flow-to-capacity ratio shown")
    print()
    print("RIGHT PANEL - Convergence Plot:")
    print("  - Blue line: Objective function trajectory")
    print("  - Red dot: Current iteration marker")
    print("  - Log scale for large value ranges")
    print("  - Shows current objective value")
    print()
    print("Watch how congestion redistributes and the objective improves")
    print("as the algorithm converges toward equilibrium/optimum!")
    print("="*70)


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('./results', exist_ok=True)

    run_animation_demo()
