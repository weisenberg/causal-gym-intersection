# Observation Variable Mapping

The dataset now uses meaningful column names instead of generic `obs_X` indices.

## Variable Breakdown (55 Total)

| Category | Variables | Description |
| :--- | :--- | :--- |
| **Agent State** | `agent_x`, `agent_y`, `agent_vx`, `agent_vy`, `agent_heading` | Position, Velocity, and Heading of the ego vehicle |
| **Lidar Sensor** | `lidar_0` to `lidar_15` | 16 ray distances detecting obstacles around the agent |
| **Nearest Cars** | `car1_rel_x`...`car5_rel_vy` | 5 nearest cars × 4 features each (Rel X, Rel Y, Rel VX, Rel VY) |
| **Nearest Pedestrians** | `ped1_rel_x`...`ped5_rel_y` | 5 nearest pedestrians × 2 features each (Rel X, Rel Y) |
| **Traffic Light** | `tl_red`, `tl_yellow`, `tl_green`, `tl_ttc` | Traffic light status and time-to-change |

## Detailed Mapping

### Agent State (5 variables)
- `agent_x`: Agent X Position
- `agent_y`: Agent Y Position
- `agent_vx`: Agent X Velocity
- `agent_vy`: Agent Y Velocity
- `agent_heading`: Agent Heading (radians)

### Lidar (16 variables)
- `lidar_0` to `lidar_15`: Distance to nearest obstacle along 16 radial rays.

### Nearest Cars (20 variables)
For each of the 5 nearest cars (e.g., `car1`):
- `car1_rel_x`: Relative X Position
- `car1_rel_y`: Relative Y Position
- `car1_rel_vx`: Relative X Velocity
- `car1_rel_vy`: Relative Y Velocity
(Repeats for `car2` through `car5`)

### Nearest Pedestrians (10 variables)
For each of the 5 nearest pedestrians (e.g., `ped1`):
- `ped1_rel_x`: Relative X Position
- `ped1_rel_y`: Relative Y Position
(Repeats for `ped2` through `ped5`)

### Traffic Light (4 variables)
- `tl_red`: Is Red? (1.0 or 0.0)
- `tl_yellow`: Is Yellow? (1.0 or 0.0)
- `tl_green`: Is Green? (1.0 or 0.0)
- `tl_ttc`: Normalized Time to Change

## Causal Discovery Analysis (FCI)

The optimized FCI run analyzes a subset of these variables (plus actions and hidden confounders).

**Analyzed Variables (21 total):**
1. `action_accel`
2. `action_steer`
3. `reward`
4. `done`
5. `temperature`
6. `traffic_density`
7. `pedestrian_density`
8. `driver_impatience`
9. `npc_color`
10. `npc_size`
11. `roughness`
12. `agent_x`
13. `agent_y`
14. `agent_vx`
15. `agent_vy`
16. `agent_heading`
17. `lidar_0`
18. `lidar_1`
19. `lidar_2`
20. `lidar_3`
21. `lidar_4`
