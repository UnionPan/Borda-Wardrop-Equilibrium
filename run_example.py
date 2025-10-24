
import numpy as np
import matplotlib.pyplot as plt
from env.traffic_model import NonAtomicTrafficEnvironment
from algorithms.frank_wolfe import FrankWolfe
from algorithms.exp_weight import ExpWeight
from algorithms.system_optimum import SystemOptimum

def run_and_plot(algorithm_class, env, **kwargs):
    """Runs an algorithm and plots its convergence."""
    print(f"Running {algorithm_class.__name__}...")
    algo = algorithm_class(env, **kwargs)
    results = algo.run()
    fig = algo.plot_convergence()
    network_name = env.loader.network_name  # FIX: correct attribute
    plt.suptitle(f'{algorithm_class.__name__} Convergence on {network_name}', fontsize=16, y=1.02)
    save_path = f'./results/{algorithm_class.__name__}_{network_name}_convergence.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence plot to {save_path}")
    return results

def main():
    """Main function to run the Sioux Falls experiment."""
    # Load the Sioux Falls network
    network_path = "./networks/SiouxFalls"
    try:
        env = NonAtomicTrafficEnvironment(network_path=network_path)
    except Exception as e:
        print(f"Error loading network: {e}")
        return

    # Run Frank-Wolfe for User Equilibrium
    run_and_plot(FrankWolfe, env, max_iterations=100, tolerance=1e-3)

    # Run ExpWeight
    run_and_plot(ExpWeight, env, max_iterations=100, learning_rate=3.63)

    # Run System Optimum
    run_and_plot(SystemOptimum, env, max_iterations=1000, tolerance=1e-4)

if __name__ == "__main__":
    main()
