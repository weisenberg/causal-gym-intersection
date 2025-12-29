
import gymnasium as gym
import numpy as np
from gym_causal_intersection.envs.simple_causal_env import SimpleCausalIntersectionEnv

def test_dims():
    import sys
    print(f"Module File: {sys.modules['gym_causal_intersection.envs.simple_causal_env'].__file__}")
    env = SimpleCausalIntersectionEnv(render_mode=None)
    print(f"Obs Space: {env.observation_space.shape}")
    
    obs, _ = env.reset()
    print(f"Reset Obs Shape: {obs.shape}")
    
    # Internal Debugging of _get_obs
    print("--- Internal State Breakdown ---")
    s = env._get_obs()
    print(f"Total State Length: {len(s)}")
    
    # Let's inspect vector components by recreating logic
    state = []
    
    # 1. Agent (4)
    state.extend(env._agent_velocity / 20.0)
    state.extend([np.cos(env._agent_heading), np.sin(env._agent_heading)])
    print(f"After Agent: {len(state)}")
    
    # 2. Extra (2)
    state.append(0) 
    state.append(0)
    print(f"After Extra: {len(state)}")
    
    # 3. Road (10)
    # 5 spurious points
    for i in range(5):
        state.extend([0,0])
    print(f"After Road: {len(state)}")
    
    # 4. NPCs (20)
    for i in range(5):
        state.extend([0,0,0,0])
    print(f"After NPCs: {len(state)}")
    
    # 5. Peds (60)
    for i in range(30):
        state.extend([0,0])
    print(f"After Peds: {len(state)}")
    
    # 6. Lane (4)
    state.extend([0,0,0,0])
    print(f"After Lane: {len(state)}")
    
    # 7. Light (4)
    state.extend([0,0,0,0])
    print(f"After Light: {len(state)}")
    
    # 8. Lidar
    l, _ = env._compute_multiray_lidar()
    print(f"Lidar Length: {len(l)}")
    state.extend(l)
    print(f"Final Count: {len(state)}")

if __name__ == "__main__":
    test_dims()
