"""Test script to verify single car and pedestrian crossing behavior."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_single_car():
    print("Testing single car environment with pedestrian crossings...")
    env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    # Test multiple resets to see car at different positions
    print("\nTesting car spawn positions (should be on right side of roads):")
    for i in range(5):
        obs, info = env.reset(seed=100 + i)
        print(f"  Reset {i+1}: Car at ({obs[0]:.0f}, {obs[1]:.0f}), heading: {obs[4]:.2f}")
    
    # Test pedestrian crossing behavior
    print("\nTesting pedestrian crossing (higher spawn rate for visibility):")
    obs, info = env.reset(seed=200, options={
        "pedestrian_spawn_rate": 0.15,
        "num_pedestrians": 10,
        "jaywalk_probability": 0.2
    })
    
    crossing_users = 0
    jaywalkers = 0
    
    print("Running 100 steps to observe pedestrians crossing...")
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        for ped in env.unwrapped.pedestrians:
            if ped["is_jaywalking"]:
                jaywalkers += 1
            else:
                crossing_users += 1
        
        if (i + 1) % 25 == 0:
            num_peds = info.get('num_pedestrians', 0)
            print(f"  Step {i+1}: {num_peds} pedestrians active")
        
        if terminated:
            print(f"  Episode terminated at step {i+1}")
            break
    
    total = crossing_users + jaywalkers
    if total > 0:
        print(f"\nPedestrian statistics:")
        print(f"  Using crossings: {crossing_users} ({crossing_users*100/total:.1f}%)")
        print(f"  Jaywalking: {jaywalkers} ({jaywalkers*100/total:.1f}%)")
    
    env.close()
    print("\n✓ Single car environment test completed!")

if __name__ == "__main__":
    test_single_car()

