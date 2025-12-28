# Gym Causal Intersection

A Gymnasium environment for simulating urban traffic scenarios, designed for **Causal Reinforcement Learning (CRL)** research. It features autonomous traffic, pedestrians, traffic lights, and configurable causal confounders (weather, traffic density, etc.).

## Environments

### 1. SimpleCausalIntersection-v0 (Curved Road Edition)
A procedural "infinite" spline-based road environment designed for robust driving and causal reasoning.
- **Goal**: Drive as far as possible along the curved lane without crashing or violating traffic rules.
    - **Features**:
      - **Procedural Spline Road**: The road is generated using splines, creating challenging curves.
      - **Infinite Runner**: Continuous driving loop.
      - **"Move or Die" Logic**:
        - **Strict Termination**: If the agent's speed drops below **2.0 m/s** for **50+ steps** (and not waiting for a red light), the episode **terminates immediately** with a **-50.0** penalty.
        - **Anti-Camping**: Forces the agent to drive to survive.
      - **"Flow or Fail" Reward v2**:
        - **Cost of Living**: **-0.1** per step constant penalty.
        - **Conditional Rewards**: 
          - **Safe**: +1.0 * Speed (Reward for moving).
          - **Blocked (Red Light)**: +1.0 * (1-Speed) (Reward for stopping).
        - **Steering Stability**: Penalty for rapid steering changes to prevent wobbling.
      - **Periodic Traffic Lights**: Implemented at fixed waypoints (Indices 100, 300).
      - **Dynamic Physics**: Lidar range and braking efficiency are affected by `temperature`.
      - **Observation Space**: 107-dimensional vector including:
        - **Lookahead CTE**: Uses a point 10m ahead for smoother steering control.
        - **Lidar**: 8 Rays with dynamic range.
        - **Traffic Light**: State & Distance.

### 2. UrbanCausalIntersection-v0
The original complex environment with a 4-way intersection.

## Installation

```bash
pip install -e .
```

## Quick Start

### Interactive Demo
```bash
python demo_simple_env.py
```

### Environments & Physics (Arcade Edition)
Recently updated with "Arcade-ified" physics for better RL training:
- **Buffed Controls**: Sharper steering (0.2), stronger brakes (2.0), and higher friction (0.8).
- **Pedestrian Horror**:
  - **Negative Lidar**: Pedestrians appear as negative values in Lidar to distinguish them from cars.
  - **Pre-Crash Fear**: Penalty (-1.0) for getting too close to pedestrians at high speed.
  - **Panic Brake**: Action 5 triggers a -1.0 full braking force.

### Training Agents
We provide scripts to train **PPO**, **DQN**, and **Dueling DQN** agents.

**1. Dueling DQN (Recommended for Stability):**
```bash
python train_dueling_dqn.py
```
- **Features**: Gradient Clipping (1.0), Dueling Architecture, Custom Logging.

**2. Standard DQN (Refined):**
```bash
python train_dqn_viz.py
```
- **Configuration**:
    - **Linear Schedule**: LR decays from 1e-4 to 0.0.
    - **Exploration**: Decays from 1.0 to 0.01 over 40% of training.
    - **Gradient Clipping**: Enabled (1.0).
    - **Safety Shield**: Force-overrides actions if `TTC < 1.5s`.

**3. PPO (Refined):**
```bash
python train_viz.py
```
- **Configuration**:
    - **Linear Schedule**: LR decays from 3e-4 to 0.0.
    - **Entropy**: Fixed at 0.0 to prevent jitter.
    - **Clip Range**: 0.2.
    - **Convergence**: Stops automatically if no improvement for 100k steps.

### Safety Features
- **Hardcoded Safety Shield**: A `SafetyWrapper` intercepts actions. If collision is imminent (TTC < 1.5s), it forces a **Panic Brake** (-1.0), overriding the agent's choice.
- **Save Best Model**: Scripts use `EvalCallback` to save the best model to `./logs/best_model_dqn/`.

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
