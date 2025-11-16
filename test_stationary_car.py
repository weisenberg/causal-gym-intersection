"""Test to verify car stays stationary and pedestrians cross properly."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_stationary_car_and_pedestrians():
    print("Testing stationary car and pedestrian crossing...")
    env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    # Test 1: Car stays stationary
    print("\n1. Testing car stays stationary:")
    obs, info = env.reset(seed=42)
    initial_pos = obs[0:2].copy()
    initial_vel = obs[2:4].copy()
    
    print(f"   Initial position: ({initial_pos[0]:.0f}, {initial_pos[1]:.0f})")
    print(f"   Initial velocity: ({initial_vel[0]:.2f}, {initial_vel[1]:.2f})")
    
    # Take 10 idle steps
    for i in range(10):
        obs, _, _, _, _ = env.step(0)  # Idle action
    
    final_pos = obs[0:2]
    final_vel = obs[2:4]
    print(f"   After 10 idle steps:")
    print(f"   Position: ({final_pos[0]:.0f}, {final_pos[1]:.0f})")
    print(f"   Velocity: ({final_vel[0]:.2f}, {final_vel[1]:.2f})")
    
    if np.allclose(initial_pos, final_pos, atol=0.1) and np.allclose(final_vel, [0, 0], atol=0.01):
        print("   ✓ Car stays stationary")
    else:
        print("   ✗ Car moved (should stay stationary)")
    
    # Test 2: Pedestrians cross from one side to the other
    print("\n2. Testing pedestrian crossing:")
    obs, info = env.reset(seed=100, options={
        'pedestrian_spawn_rate': 0.2,
        'num_pedestrians': 3
    })
    
    crossing_observed = False
    for i in range(150):
        obs, _, _, _, info = env.step(0)  # Car stays idle
        
        for ped in env.unwrapped.pedestrians:
            # Check if pedestrian crosses (starts on one side, ends on opposite)
            if ped["crossed"] and not ped["is_jaywalking"]:
                crossing_observed = True
                if i % 30 == 0:
                    print(f"   Step {i}: Pedestrian at ({ped['pos'][0]:.0f}, {ped['pos'][1]:.0f}), "
                          f"target: ({ped['target'][0]:.0f}, {ped['target'][1]:.0f}), "
                          f"crossed crossing: {ped['crossed']}")
    
    if crossing_observed:
        print("   ✓ Pedestrians cross from one sidewalk to the other")
    else:
        print("   ✗ Pedestrians not crossing properly")
    
    env.close()
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    test_stationary_car_and_pedestrians()

