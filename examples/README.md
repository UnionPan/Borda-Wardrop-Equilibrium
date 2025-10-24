# Flow Animation Examples

This directory contains scripts for creating animated visualizations of traffic flow evolution.

## Scripts

### `quick_animation_demo.py`
**Quick test** - Creates a single animation with 50 iterations (~10 frames).

```bash
python examples/quick_animation_demo.py
```

**Output**: `./results/quick_flow_animation.gif` (~1-2 minutes to generate)

### `animate_flow_evolution.py`
**Full demo** - Creates three animations:
1. ExpWeight (365 iterations) - "365 day" simulation
2. Frank-Wolfe (100 iterations)
3. System Optimum (100 iterations)

```bash
python examples/animate_flow_evolution.py
```

**Outputs**:
- `./results/expweight_365day_flow_animation.gif` (73 frames, 10 fps)
- `./results/frankwolfe_flow_animation.gif` (20 frames, 5 fps)
- `./results/system_optimum_flow_animation.gif` (20 frames, 5 fps)

**Time**: ~15-30 minutes total depending on your system.

## What do the animations show?

Each frame shows:
- **OpenStreetMap background** - Real geographic context
- **Colored links** - Red = high congestion, Yellow = moderate, White = light traffic
- **Iteration number** - Algorithm progress
- **Objective value** - Beckmann potential or System Cost

Watch how congestion "hotspots" shift and evolve as the algorithm converges!

## Requirements

- **Pillow** (PIL): For GIF creation
  ```bash
  pip install Pillow
  ```

- **Geospatial libraries** (for OSM background):
  ```bash
  pip install osmnx geopandas contextily
  ```

## Custom animations

To create your own animation:

```python
from env.traffic_model import NonAtomicTrafficEnvironment
from algorithms.exp_weight import ExpWeight
from utils.visualization import animate_flow_evolution

# Load network
env = NonAtomicTrafficEnvironment("./networks/SiouxFalls")

# Run algorithm with full array storage
alg = ExpWeight(env, max_iterations=100)
alg.store_full_arrays = True  # REQUIRED for animations!
alg.run()

# Create animation
animate_flow_evolution(
    env,
    alg,
    output_path='my_animation.gif',
    fps=5,              # Frames per second
    interval_frames=5,  # Show every 5th iteration
    figsize=(16, 12)    # Figure size
)
```

## Tips

- **Faster generation**: Increase `interval_frames` (e.g., 10 instead of 5)
- **Smoother animation**: Decrease `interval_frames` (e.g., 2 or 1)
- **Speed up playback**: Increase `fps`
- **Slow down playback**: Decrease `fps`
- **Larger images**: Increase `figsize`

Note: More frames = longer generation time!
