# Gym Causal Intersection

A Gymnasium environment for simulating urban traffic intersections with cars, pedestrians, and traffic lights. This environment is designed for reinforcement learning research, particularly for causal reinforcement learning applications.

## Overview

The environment simulates a 2D top-down urban intersection where an agent (a red car) must navigate safely while obeying traffic rules. The environment includes:

- **Agent Vehicle**: A red car controlled by the RL agent
- **NPC Cars**: Autonomous vehicles with varied attributes (speed, size, color) that follow traffic rules
- **Pedestrians**: Blue pedestrians using zebra crossings and red jaywalkers crossing streets
- **Traffic Lights**: Directional traffic lights at intersections with green/yellow/red phases
- **Zebra Crossings**: Marked pedestrian crossings at intersections

## Environment Specifications

### Map Dimensions

- **Window Size**: 600×600 pixels
- **Road Width**: 100 pixels (roads span from 250 to 350 pixels in both x and y)
- **Intersection Size**: 120×120 pixels (centered at 300, 300)
- **Road Center**: 300 pixels (both horizontal and vertical roads)

### Car Dimensions and Rendering

#### Agent Car (Red)
- **Length**: 20 pixels
- **Width**: 10 pixels
- **Center**: The car's position (`_agent_location`) represents the **center** of the car
- **Rendering**: The car is drawn as a rotated rectangle (polygon) centered at `_agent_location`
  - Corners are calculated relative to center: `[-length/2, -width/2]` to `[length/2, width/2]`
  - Rotation matrix accounts for pygame's y-down coordinate system
  - Heading: 0 = right (east), π/2 = up (north), π = left (west), -π/2 = down (south)

#### NPC Cars
- **Length**: Variable (default 20, can range from 18-25 pixels)
- **Width**: Variable (default 10, can range from 8-15 pixels)
- **Center**: Each NPC car's `pos` represents the **center** of the car
- **Rendering**: Same as agent car, but with varied colors (rainbow shades)
- **Attributes**: Each NPC car has individual:
  - `cruise_speed`: Target speed (2.5-4.5 pixels/step)
  - `accel`: Acceleration rate (0.2-0.3)
  - `brake_factor`: Braking efficiency (0.85-0.95)
  - `stopping_distance`: Distance at which to start braking (30-50 pixels)
  - `color`: RGB tuple (rainbow shades)

**Important**: All car positions (`_agent_location` for agent, `car["pos"]` for NPCs) represent the **geometric center** of the vehicle. Collision detection uses bounding circles with radius `max(length, width) / 2.0` to account for the full car footprint.

### Right-Side Driving Rule

NPC cars follow right-side driving rules based on their direction:

- **Eastbound** (left to right): Bottom side of horizontal road (y = 330)
- **Westbound** (right to left): Upper side of horizontal road (y = 270)
- **Northbound** (bottom to top): Right side of vertical road (x = 320)
- **Southbound** (top to bottom): Left side of vertical road (x = 280)

This prevents head-on collisions and ensures realistic traffic flow.

## Action Space

The action space is **continuous** with 2 dimensions:

```python
action_space = Box(low=[-1.0, -1.0], high=[1.0, 1.0], dtype=np.float32)
```

- **Action[0]**: Acceleration command
  - `1.0`: Accelerate forward
  - `-1.0`: Brake (until stop, no reverse)
  - `0.0`: No acceleration/braking
- **Action[1]**: Steering command
  - `1.0`: Turn left (counterclockwise)
  - `-1.0`: Turn right (clockwise)
  - `0.0`: No steering

**Physics**:
- Acceleration: `0.3` pixels/step²
- Angular velocity: `0.1` radians/step
- Max speed: `5.0` pixels/step
- No-slide physics: Velocity vector aligns with heading when moving
- Friction: Applied when moving (default `1.0`, no decay)

**Braking Behavior**: When braking (action[0] < 0), the car applies negative acceleration but **cannot reverse**. If velocity would go negative along the heading direction, it's clamped to zero.

## Observation Space

The observation space is **configurable** via `observation_type` parameter:

### Kinematic Observations (Default)

```python
observation_space = Box(low=-inf, high=inf, shape=(55,), dtype=np.float32)
```

The 55-dimensional vector concatenates:

1. **Agent State** (5 dims):
   - `[pos_x, pos_y, vel_x, vel_y, heading]`

2. **LIDAR Sensors** (16 dims):
   - `[dist_0, dist_1, ..., dist_15]`
   - 16 rays evenly spaced in 360° around the agent
   - Max range: 200 pixels
   - Detects NPC cars, pedestrians, and map boundaries

3. **Nearest k=5 Cars** (20 dims):
   - For each of the 5 nearest NPC cars:
   - `[rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]`
   - Relative position and velocity to agent
   - Padded with zeros if fewer than 5 cars exist

4. **Nearest k=5 Pedestrians** (10 dims):
   - For each of the 5 nearest pedestrians:
   - `[rel_pos_x, rel_pos_y]`
   - Relative position to agent
   - Padded with zeros if fewer than 5 pedestrians exist

5. **Next Traffic Light** (4 dims):
   - `[is_red, is_yellow, is_green, time_to_change]`
   - One-hot encoding of light state for agent's direction
   - Time to change: normalized remaining time in current phase

### Pixel Observations

```python
observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
```

- RGB array from the render function, resized to 84×84
- Top-down view of the entire 600×600 map
- Includes all visual elements (roads, cars, pedestrians, traffic lights, zebra crossings)

## Reward Function

The reward is **dense and shaped**, computed and applied **at every step** as the agent moves around. The total reward for each step is the sum of multiple components:

### Positive Rewards (Per Step)

- **Survival Reward**: `+0.1` **every step** (encourages staying alive)
- **Progress/Efficiency Reward**: `+1.0 * (agent_speed / max_speed)` **every step** (encourages forward movement)
  - This reward scales with speed: faster movement = higher reward
  - At max speed (5.0), this gives `+1.0` per step
  - At half speed (2.5), this gives `+0.5` per step
  - When stationary, this gives `+0.0`

### Positive Rewards (Termination Only)

- **Successful Exit**: `+10.0` when agent leaves the map without collision (episode terminates)

### Negative Rewards (Per Step)

- **Red Light Violation**: `-0.5` **every step** while running a red light
- **Safety Buffer Proximity**: `-0.1` **every step** while within `safety_buffer` distance of an NPC car
- **Zebra Crossing with Pedestrian**: `-2.0` **every step** while agent is on a zebra crossing with a pedestrian on it

### Negative Rewards (One-Time Events)

- **Hitting NPC Car**: `-10.0` when collision occurs (episode continues, penalty applied once per collision)
- **Crash Penalty**: `-100.0` when hitting a pedestrian/jaywalker (episode terminates - worst case)

### Reward Calculation Example

At each step, the reward is calculated as:
```
reward = +0.1                                    # Survival
       + 1.0 * (speed / max_speed)               # Progress (0.0 to 1.0)
       - 0.5 if running_red_light                 # Red light penalty
       - 0.1 if near_npc_car                     # Proximity penalty
       - 2.0 if on_crossing_with_pedestrian       # Crossing penalty
       - 10.0 if hit_npc_car                      # NPC collision (one-time)
       - 100.0 if hit_pedestrian                  # Pedestrian collision (one-time, terminates)
       + 10.0 if off_screen_safely                # Success bonus (one-time, terminates)
```

**Note**: The survival and progress rewards are given **every step**, encouraging continuous safe movement. The agent accumulates rewards as it navigates the environment.

### Termination Conditions

The episode terminates when:
1. **Pedestrian Collision**: Agent hits a pedestrian or jaywalker → `-100.0` reward
2. **Off-Screen**: Agent leaves the map boundaries → `+10.0` reward (success)
3. **Time Limit Success**: Agent survives for 59 seconds (1770 steps at 30 FPS) without collision → `info["success"] = True`

## NPC Car Collision Avoidance

NPC cars implement sophisticated collision avoidance to prevent deadlocks and ensure realistic behavior:

### Detection Logic

1. **Initialization**: Each step, NPC cars assume the front is clear (`target_speed = cruise_speed`)

2. **Traffic Light Check**: 
   - Stop before intersection if light is red/yellow
   - Distance: `stopping_distance` (30-50 pixels) before intersection edge
   - If inside intersection, clear it regardless of light state

3. **Obstacle Detection (Front-Only)**:
   - **Agent Car**: Only react if agent is ahead within 60° cone (`cos_angle > 0.5` and `dist_along > 10`)
   - **Pedestrians**: Only react if pedestrian is ahead within 60° cone (front-only, not sides/back)
   - **Other NPC Cars**: Only react if other car is ahead within 60° cone (front/back alignment for cars)

4. **Speed Adjustment**:
   - **Immediate Stop**: If obstacle within `min_margin` (car width + 6 pixels)
   - **Slow Down**: If obstacle within `stopping_distance`:
     - Within 60% of stopping distance: `target_speed = 20% of cruise_speed`
     - Within stopping distance: `target_speed = 50% of cruise_speed`

5. **Deadlock Prevention**:
   - If no obstacles detected in front and not stopped at light → `target_speed = cruise_speed`
   - Cars continue moving when front is clear, preventing intersection deadlocks

### Final Collision Resolver

After physics update, a final separation pass ensures no geometric overlap:

- **Car-to-Car**: Uses `max(length, width) / 2.0` as bounding radius
  - Only resolves if cars are aligned front/back (`abs(cos_angle) > 0.5`)
  - Pushes cars apart and sets `target_speed = 0.0` (can resume next step if clear)

- **Car-to-Pedestrian**: Uses car's bounding radius + pedestrian radius
  - Only resolves if pedestrian is in front (`cos_angle > 0.5` and `dist_along > 10`)
  - Pushes car away and stops it

- **Stopped Cars**: If a car is stopped (speed < 0.1 and target_speed < 0.1), it **cannot be pushed** by cars behind it. Only the moving car gets repositioned.

## Traffic Lights

### Directional Traffic Lights

Each intersection has **4 directional traffic lights** (one per approach: north, south, east, west):

- **4-Phase Cycle**:
  1. Phase 0: North-South green (30 steps), East-West red
  2. Phase 1: North-South yellow (5 steps), East-West red
  3. Phase 2: East-West green (30 steps), North-South red
  4. Phase 3: East-West yellow (5 steps), North-South red

- **Synchronization**: Lights are synchronized in pairs (2×2):
  - North-South: Synchronized (both same state)
  - East-West: Synchronized (both same state)
  - Perpendicular pairs alternate (when N-S is green, E-W is red)

### Pedestrian Traffic Light Interaction

- Pedestrians wait for **perpendicular traffic** to be red before crossing
- If a pedestrian is already on a crossing when the light turns green, they **continue crossing** to completion
- This prevents pedestrians from being stranded in the middle of the road

## Pedestrians

### Types

1. **Regular Pedestrians** (Blue):
   - Spawn at intersection corners
   - Always use zebra crossings
   - Follow traffic light rules (wait for perpendicular traffic to be red)

2. **Jaywalkers** (Red):
   - Spawn on road sides (not at intersections)
   - Cross streets directly (perpendicular to road)
   - Do not use zebra crossings
   - Do not follow traffic lights

### Behavior

- **Crossing Phases**:
  1. `to_crossing`: Move to zebra crossing start
  2. `on_crossing`: Move along zebra crossing (hard constraint keeps them on the line)
  3. `to_destination`: Move to destination corner

- **Hard Constraint**: Regular pedestrians are forced to stay exactly on the zebra crossing line during the `on_crossing` phase

- **Speed**: `2.0` pixels/step

- **Radius**: `5` pixels (for collision detection)

## Usage

### Basic Usage

```python
import gymnasium as gym
import gym_causal_intersection

# Create environment
env = gym.make('UrbanCausalIntersection-v0', render_mode='human')

# Reset
obs, info = env.reset(seed=42)

# Step
action = np.array([1.0, 0.0])  # Accelerate forward, no steering
obs, reward, terminated, truncated, info = env.step(action)

# Close
env.close()
```

### Configuration

```python
# With kinematic observations (default)
env = gym.make('UrbanCausalIntersection-v0', observation_type='kinematic')

# With pixel observations
env = gym.make('UrbanCausalIntersection-v0', observation_type='pixel')

# With custom context
obs, info = env.reset(seed=42, options={
    'pedestrian_spawn_rate': 0.05,
    'num_pedestrians': 8,
    'jaywalk_probability': 0.15,
    'traffic_light_duration': 30
})
```

### Interactive Demo

Run the demo script for manual control:

```bash
python demo_env.py
```

**Controls** (WASD):
- **W**: Accelerate forward
- **S**: Brake (until stop)
- **A**: Turn left
- **D**: Turn right
- **ESC**: Close environment

## Extended Environment

There is also an extended environment (`UrbanCausalIntersectionExtended-v0`) with:

- **Larger Map**: Configurable size (default 2000×2000 pixels)
- **Multiple Intersections**: 3-way and 4-way intersections
- **More Roads**: Longer streets connecting intersections
- **Camera Following**: Camera follows agent in larger map
- **Same Features**: All features from simple environment (NPC cars, pedestrians, traffic lights, etc.)

## Technical Details

### Coordinate System

- **Origin**: Top-left corner (0, 0)
- **X-axis**: Increases rightward (0 to 600)
- **Y-axis**: Increases downward (0 to 600) - **pygame convention**
- **Heading**: 0 = right (east), π/2 = up (north), π = left (west), -π/2 = down (south)

### Physics

- **Velocity**: 2D vector `[vx, vy]` in pixels/step
- **Heading**: Angle in radians (0 to 2π)
- **No-Slide**: When moving, velocity vector is aligned with heading direction
- **Friction**: Applied per step (default `1.0` = no decay)

### Collision Detection

- **Agent-to-Pedestrian**: Circle-to-circle (agent: `car_width/2`, pedestrian: `pedestrian_radius`)
- **Agent-to-NPC-Car**: Circle-to-circle (both use `max(length, width)/2.0` as radius)
- **NPC-to-NPC**: Circle-to-circle with directional checks (front/back only)
- **NPC-to-Pedestrian**: Circle-to-circle with directional checks (front-only)

### Rendering

- **Framework**: Pygame
- **Frame Rate**: 30 FPS
- **Background**: Gray (128, 128, 128)
- **Roads**: Dark gray (64, 64, 64)
- **Intersections**: Medium gray (80, 80, 80)
- **Agent Car**: Red (255, 0, 0)
- **NPC Cars**: Rainbow shades (varied)
- **Pedestrians**: Blue (100, 100, 255) for regular, Red (255, 100, 100) for jaywalkers
- **Traffic Lights**: Green (0, 255, 0), Yellow (255, 255, 0), Red (255, 0, 0)

## File Structure

```
gym-causal-intersection/
├── gym_causal_intersection/
│   ├── __init__.py
│   └── envs/
│       ├── causal_intersection_env.py          # Simple environment
│       └── causal_intersection_extended_env.py # Extended environment
├── demo_env.py                                  # Interactive demo
├── demo_extended_env.py                         # Extended demo
├── test_env.py                                  # Unit tests
└── README.md                                    # This file
```

## License

[Specify your license here]

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{gym_causal_intersection,
  title={Gym Causal Intersection: A Reinforcement Learning Environment for Urban Traffic},
  author={[Your Name]},
  year={2024},
  url={[Your Repository URL]}
}
```
