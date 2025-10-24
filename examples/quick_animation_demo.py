"""
Quick Animation Demo - Single Algorithm

Creates a quick flow animation showing congestion evolution.
Much faster than the full 365-day demo.
"""

import sys
sys.path.insert(0, '.')

from env.traffic_model import NonAtomicTrafficEnvironment
from algorithms.exp_weight import ExpWeight
from utils.visualization import animate_flow_evolution
import os


def main():
    """Run a quick animation demo with 50 iterations."""

    print("Quick Flow Animation Demo")
    print("="*60)
    print()

    # Create results directory
    os.makedirs('./results', exist_ok=True)

    # Load network
    print("Loading Sioux Falls network...")
    env = NonAtomicTrafficEnvironment("./networks/SiouxFalls")
    print(f"  {env.num_links} links, {env.num_paths} paths")
    print()

    # Run ExpWeight with 50 iterations
    print("Running ExpWeight algorithm (50 iterations)...")
    ew = ExpWeight(
        env,
        max_iterations=50,
        learning_rate=0.1,
        tolerance=1e-5
    )

    # CRITICAL: Must enable full array storage for animations!
    ew.store_full_arrays = True

    results = ew.run(verbose=True)
    print()

    print(f"Results:")
    print(f"  Beckmann improved from {ew.history['beckmann_potential'][0]:.2e}")
    print(f"                      to {results['final_beckmann']:.2e}")
    print(f"  {ew.history['beckmann_potential'][0] / results['final_beckmann']:.1f}x better!")
    print()

    # Create animation
    print("Creating enhanced flow animation...")
    print("  - Side-by-side: Network map + Convergence plot")
    print("  - Showing every 5th iteration (10 frames total)")
    print("  - High contrast colors (white -> yellow -> red -> black)")
    print("  - Thicker lines = higher congestion")
    print()

    animate_flow_evolution(
        env,
        ew,
        output_path='./results/quick_flow_animation.gif',
        fps=3,  # 3 frames per second (slower to see changes)
        interval_frames=5,  # Every 5th iteration
        figsize=(20, 10),  # Wider for side-by-side layout
        show_convergence_plot=True  # Show Beckmann trajectory
    )

    print()
    print("✓ Animation saved to: ./results/quick_flow_animation.gif")
    print()
    print("The animation shows:")
    print("  LEFT:  Network with flow visualization")
    print("         - White/Yellow = low congestion")
    print("         - Orange/Red = moderate congestion")
    print("         - Dark Red/Black = severe congestion")
    print("         - Line thickness = congestion level")
    print()
    print("  RIGHT: Beckmann potential decreasing over time")
    print("         - Blue line = convergence trajectory")
    print("         - Red dot = current iteration")
    print()
    print("Watch how congestion redistributes as the algorithm converges!")


if __name__ == "__main__":
    main()
