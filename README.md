# Gym Causal Intersection

A Gymnasium environment for simulating urban traffic scenarios, designed for **Causal Reinforcement Learning (CRL)** research. It features autonomous traffic, pedestrians, traffic lights, and configurable causal confounders (weather, traffic density, etc.).

## Environments

### 1. SimpleCausalIntersection-v0 (New & Recommended)
A simplified, "infinite" vertical road environment designed for faster training and clear causal analysis.
- **Goal**: Drive as far north/south as possible without crashing.
- **Features**:
  - **Infinite Road**: The road extends indefinitely (visually loops/extends), allowing for long-distance driving.
  - **Randomized Layout**: At each reset, the road's rotation and position are randomized, forcing the agent to generalize rather than memorize coordinates.
  - **Randomized Spawn**: Agent spawns with a random heading offset ($\pm 28^\circ$), requiring immediate steering correction.
  - **Discrete Actions**: Simplified action space (Idle, Accelerate, Brake, Left, Right).
  - **Spurious Correlations**: Visual features like NPC car color and size can be randomized to test agent robustness against non-causal factors.

### 2. UrbanCausalIntersection-v0
The original complex environment with a 4-way intersection.
- **Features**: Full traffic light cycles, turning lanes, and complex right-of-way rules.
- **Continuous Actions**: Fine-grained steering and acceleration control.

## Installation

```bash
pip install -e .
```

## Quick Start

### Interactive Demo
Manually control the car to get a feel for the physics and rules.

**Simple Environment (Recommended):**
```bash
python demo_simple_env.py
```
**Controls (WASD):**
- `W`: Accelerate
- `S`: Brake
- `A`: Steer Left
- `D`: Steer Right

### Training Agents
We provide scripts to train **PPO** and **DQN** agents on the `SimpleCausalIntersection-v0` environment.

**DQN (Best Performance):**
```bash
python train_dqn_viz.py
```
- **Results**: Converges to consistent success (reaching the goal) even with randomized layouts and starting headings.
- **Output**: Saves plots and videos to `videos_dqn_random/`.

**PPO:**
```bash
python train_viz.py
```
- **Note**: PPO currently struggles with this specific discrete/randomized setup compared to DQN.

## Causal Discovery
The project includes a pipeline to generate data and discover the underlying causal graph of the environment using the **FCI (Fast Causal Inference)** algorithm.

1.  **Generate Data**:
    Run the agent (random or trained) to collect interaction data.
    ```bash
    python generate_simple_data.py
    ```
    - Generates `simple_env_data.csv` (default: 1,000,000 steps).

2.  **Run FCI Algorithm**:
    Analyze the data to reconstruct the causal graph.
    ```bash
    python run_fci_optimized.py
    ```
    - **Output**: `causal_graph_fci_optimized.png`
    - **Outcome**: Successfully recovers the true causal links (e.g., `Traffic Light -> Agent Speed`, `Brake -> Velocity`) while correctly identifying confounders.

## Key Features for Research

### Domain Randomization
The environment supports extensive randomization to prevent overfitting and test generalization:
- **Layout**: Rotation and center position of the road.
- **Spawn Conditions**: Initial position and heading.
- **Visuals**: NPC car colors (red, blue, rainbow) and sizes.
- **Physics**: Friction and braking efficiency based on "Temperature".

### Causal Variables
The environment exposes ground-truth causal variables in the `info` dictionary, allowing for direct verification of causal discovery algorithms.
- `temperature`: Affects physics (friction).
- `traffic_density`: Affects NPC count.
- `driver_impatience`: Affects NPC acceleration/behavior.

## File Structure
- `gym_causal_intersection/envs/`: Environment source code.
  - `simple_causal_env.py`: The infinite vertical road env.
  - `causal_intersection_env.py`: Base class and 4-way intersection env.
- `train_dqn_viz.py`: Main DQN training script.
- `train_viz.py`: Main PPO training script.
- `demo_simple_env.py`: Manual control demo.
- `generate_simple_data.py`: Causal data generation.
- `run_fci_optimized.py`: Causal graph discovery.

## Citation
If you use this environment, please cite:
```bibtex
@software{gym_causal_intersection,
  title={Gym Causal Intersection: A Reinforcement Learning Environment for Urban Traffic},
  author={Ali Khadangi},
  year={2025},
  url={https://github.com/weisenberg/causal-gym-intersection}
}
```
