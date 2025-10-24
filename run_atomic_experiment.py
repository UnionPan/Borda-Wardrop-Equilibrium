"""
Example script to run population-level algorithms on an atomic environment.
"""

import matplotlib.pyplot as plt
from env.traffic_model import AtomicTrafficEnvironment
from algorithms.atomic_exp_weight import AtomicExpWeight
from algorithms.atomic_frank_wolfe import AtomicFrankWolfe


def run_and_plot(algorithm, title_suffix: str, save_suffix: str):
    """Helper to run an algorithm and persist its convergence plot."""
    algo = algorithm
    algo.run()
    fig = algo.plot_convergence()
    plt.suptitle(
        f"{algo.__class__.__name__} Convergence on {title_suffix}",
        fontsize=16,
        y=1.02
    )
    save_path = f'./results/{algo.__class__.__name__}_{save_suffix}_convergence.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence plot to {save_path}")


def main():
    """Main function to run the population-level atomic experiment."""
    network_path = "./networks/Braess-Example"
    background_ratio = 0.5

    # ExpWeight dynamics (Adam style) for strategic populations
    try:
        env_exp = AtomicTrafficEnvironment(
            network_path=network_path,
            background_ratio=background_ratio
        )
    except Exception as exc:
        print(f"Error loading network for ExpWeight: {exc}")
        return

    exp_algo = AtomicExpWeight(env_exp, max_iterations=100, learning_rate=0.05)
    run_and_plot(exp_algo, "Braess-Example", "braess")

    # Frank-Wolfe on the same environment class (new instance for a clean slate)
    try:
        env_fw = AtomicTrafficEnvironment(
            network_path=network_path,
            background_ratio=background_ratio
        )
    except Exception as exc:
        print(f"Error loading network for Frank-Wolfe: {exc}")
        return

    fw_algo = AtomicFrankWolfe(env_fw, max_iterations=100, tolerance=1e-4)
    run_and_plot(fw_algo, "Braess-Example", "braess")


if __name__ == "__main__":
    main()
