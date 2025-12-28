# Gym Causal Intersection

A Gymnasium environment for simulating urban traffic scenarios, designed for **Causal Reinforcement Learning (CRL)** research. It features autonomous traffic, pedestrians, traffic lights, and configurable causal confounders (weather, traffic density, etc.).

## Environments

### 1. SimpleCausalIntersection-v0 (Curved Road Edition)
A procedural "infinite" spline-based road environment designed for robust driving and causal reasoning.
- **Goal**: Drive as far as possible along the curved lane without crashing or violating traffic rules.
- **Features**:
  - **Procedural Spline Road**: The road is generated using splines, creating challenging curves instead of a straight line.
  - **Infinite Runner**: When the agent reaches the end, it is teleported back to the start (preserving momentum) for continuous training.
  - **Periodic Traffic Lights**: Implemented at fixed waypoints (Indices 100, 300).
    - **State Machine**: Cycles through Green -> Yellow -> Red.
    - **Penalties**: Running a Red Light results in a severe penalty (-50.0) and immediate episode termination.
    - **Pedestrian Sync**: Pedestrians only attempt to cross the road when the traffic light is Red for vehicles.
  - **Dynamic Physics**: Lidar range and braking efficiency are affected by the environment's `temperature` (Causal Confounder).
  - **Observation Space**: 107-dimensional vector including:
    - Agent State (Vel, Heading, CTE)
    - **Lidar**: 8 Rays with dynamic range (visualized in Yellow/Red).
    - **Traffic Light**: One-hot encoding (Green/Yellow/Red) + Distance to next light.
    - Nearby Entities (NPCs, Pedestrians).

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

**Visualization Features:**
- **UI Overlay**: Top-left stats (Episode, Step, Reward).
- **Environment Info**: Top-right stats (Light State, Dist, Temp).
- **Visuals**: 
    - **Lidar Rays**: Yellow (clear) / Red (blocked), length changes with temperature.
    - **Lane Center**: Cyan line indicating the ideal path.
    - **Traffic Light**: Colored circle indicating current state.

### Training Agents
We provide scripts to train **PPO** and **DQN** agents on the `SimpleCausalIntersection-v0` environment.

**DQN (Best Performance - Tuned):**
```bash
python train_dqn_viz.py
```
- **Configuration**:
    - **Hyperparameters**: LR `1e-4`, Buffer `200k`, Exploration `30%`, Batch `128`.
    - **Success Threshold**: Training automatically stops if Episode Reward > **800.0** (Approx. 40-50s of driving).
    - **Output**: Saves plots and videos to `videos_dqn_curved/`.

**PPO:**
```bash
python train_viz.py
```
- **Note**: PPO currently struggles with this specific discrete/randomized setup compared to DQN.

## Causal Discovery
The project includes a pipeline to generate data and discover the underlying causal graph using the **FCI (Fast Causal Inference)** algorithm.

1.  **Generate Data**:
    ```bash
    python generate_simple_data.py
    ```
    - Generates `simple_env_data.csv` (default: 1,000,000 steps).

2.  **Run FCI Algorithm**:
    ```bash
    python run_fci_optimized.py
    ```
    - **Output**: `causal_graph_fci_optimized.png`
    - **Outcome**: Successfully recovers true causal links (e.g., `Traffic Light -> Agent Speed`).

## Key Features for Research

### Domain Randomization & Causal Variables
- **Layout**: Rotation and center position of the road.
- **Visuals**: NPC car colors (red, blue, rainbow) and sizes.
- **Physics**: 
    - `temperature`: Affects friction and Lidar range (simulating visibility/braking conds).
    - `traffic_density`: Affects NPC count.
    - `driver_impatience`: Affects NPC acceleration.

## File Structure
- `gym_causal_intersection/envs/`: Source code.
  - `simple_causal_env.py`: The curved road env with traffic lights.
- `train_dqn_viz.py`: Main DQN training script (Tuned).
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
