"""Test script specifically for pedestrian and zebra crossing features."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_pedestrians():
    # Create environment without rendering for faster testing
    env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    print("Testing pedestrian and zebra crossing features...")
    print(f"Default context: {env.unwrapped.context}")
    
    # Reset with custom context - higher spawn rate for testing
    obs, info = env.reset(seed=42, options={
        "pedestrian_spawn_rate": 0.2,  # 20% chance per step
        "num_pedestrians": 10,
        "jaywalk_probability": 0.4
    })
    
    print(f"Custom context: {env.unwrapped.context}")
    print(f"Initial pedestrians: {info.get('num_pedestrians', 0)}")
    print(f"Zebra crossings: {len(env.unwrapped.zebra_crossings)}")
    
    # Run for more steps to see pedestrians spawn and move
    print("\nRunning 50 steps to observe pedestrians...")
    max_pedestrians = 0
    collision_occurred = False
    
    for i in range(50):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        num_peds = info.get('num_pedestrians', 0)
        max_pedestrians = max(max_pedestrians, num_peds)
        
        if num_peds > 0 and i % 10 == 0:
            print(f"Step {i}: Pedestrians: {num_peds}, Reward: {reward:.2f}")
            
            # Check pedestrian positions
            for j, ped in enumerate(env.unwrapped.pedestrians):
                ped_type = "jaywalking" if ped["is_jaywalking"] else "using crossing"
                print(f"  Pedestrian {j}: pos=({ped['pos'][0]:.1f}, {ped['pos'][1]:.1f}), {ped_type}")
        
        if terminated:
            print(f"\nEpisode terminated at step {i}")
            if info.get('collision', False):
                print("  Reason: Collision with pedestrian!")
                collision_occurred = True
            else:
                print("  Reason: Off-screen")
            break
    
    print(f"\nTest Results:")
    print(f"  Max pedestrians observed: {max_pedestrians}")
    print(f"  Collision occurred: {collision_occurred}")
    print(f"  Final pedestrians: {info.get('num_pedestrians', 0)}")
    
    env.close()
    print("\nPedestrian test completed successfully!")

if __name__ == "__main__":
    test_pedestrians()
