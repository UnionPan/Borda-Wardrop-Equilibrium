"""
Example script to run dueling-style algorithms on an atomic environment.

Uses both DuelingExpWeight (preference learning) and DuelingFrankWolfe
(stochastic migration) on the Braess example network and stores convergence
plots under ./results.
"""

import os

import matplotlib.pyplot as plt

from env.traffic_model import AtomicTrafficEnvironment
from algorithms.dueling_exp_weight import DuelingExpWeight
from algorithms.dueling_frank_wolfe import DuelingFrankWolfe


def run_and_plot(algorithm, title_suffix: str, save_suffix: str):
    """Run an algorithm and persist its convergence plot."""
    algo = algorithm
    algo.run()

    fig = algo.plot_convergence()
    plt.suptitle(
        f"{algo.__class__.__name__} Convergence on {title_suffix}",
        fontsize=16,
        y=1.02,
    )

    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/{algo.__class__.__name__}_{save_suffix}_convergence.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence plot to {save_path}")


def main():
    """Main entry point for the dueling algorithm experiment."""
    network_path = "./networks/Braess-Example"
    background_ratio = 0.5
    seed = 42

    # DuelingExpWeight: preference-based path updates
    try:
        env_exp = AtomicTrafficEnvironment(
            network_path=network_path,
            background_ratio=background_ratio,
            random_state=seed,
        )
    except Exception as exc:
        print(f"Error loading network for DuelingExpWeight: {exc}")
        return

    dueling_exp = DuelingExpWeight(
        env_exp,
        max_iterations=150,
        learning_rate=0.5,
        temperature=1.0,
        duels_per_iteration=2,
        tolerance=1e-4,
        random_state=seed,
    )
    run_and_plot(dueling_exp, "Braess-Example", "braess")

    # DuelingFrankWolfe: stochastic Frank-Wolfe migration
    try:
        env_fw = AtomicTrafficEnvironment(
            network_path=network_path,
            background_ratio=background_ratio,
            random_state=seed + 1,
        )
    except Exception as exc:
        print(f"Error loading network for DuelingFrankWolfe: {exc}")
        return

    dueling_fw = DuelingFrankWolfe(
        env_fw,
        max_iterations=150,
        tolerance=1e-4,
        step_size_method="optimal",
        random_state=seed + 1,
    )
    run_and_plot(dueling_fw, "Braess-Example", "braess")


if __name__ == "__main__":
    main()
