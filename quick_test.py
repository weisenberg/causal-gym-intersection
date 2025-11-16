"""Quick test to show the environment setup."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def quick_test():
    print("Creating UrbanCausalIntersection-v0 environment...")
    env = gym.make('UrbanCausalIntersection-v0', render_mode='human')
    
    print("\n" + "="*60)
    print("ENVIRONMENT VISUALIZATION")
    print("="*60)
    print("\nWhat you should see in the window:")
    print("  🚗 Red rectangle = Your car (spawns on right side of road)")
    print("  ⚪ Blue circles = Pedestrians using zebra crossings")
    print("  ⚪ Red circles = Jaywalking pedestrians")
    print("  ⬜ White stripes = Zebra crossings (4 crossings)")
    print("  ⬛ Dark gray = Roads (two-way traffic)")
    print("  ⬜ Light gray = Background")
    print("\n" + "="*60)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"\nEnvironment initialized!")
    print(f"  Car at: ({obs[0]:.0f}, {obs[1]:.0f})")
    print(f"  Car heading: {obs[4]:.2f} radians (facing intersection)")
    print(f"  Initial pedestrians: {info.get('num_pedestrians', 0)}")
    
    print("\nRunning 100 steps to show pedestrian spawning...")
    print("(Watch the window to see pedestrians appear and move)")
    
    for i in range(100):
        # Random actions for variety
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if (i + 1) % 20 == 0:
            num_peds = info.get('num_pedestrians', 0)
            print(f"  Step {i+1}: {num_peds} pedestrians active")
        
        if terminated:
            print(f"\n  Episode terminated at step {i+1}")
            break
    
    print("\nTest complete! Close the window to exit.")
    print("\nTo see a continuous demo, run: python demo_env.py")
    
    # Keep window open
    import time
    time.sleep(5)
    
    env.close()

if __name__ == "__main__":
    quick_test()

