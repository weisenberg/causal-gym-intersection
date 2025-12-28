import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_env(env_id):
    print(f"Testing {env_id}...")
    try:
        env = gym.make(env_id, render_mode=None)
        obs, info = env.reset()
        print(f"  Reset successful. Obs shape: {obs.shape}")
        
        # Take 10 steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
        print(f"  Stepping successful.")
        env.close()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test Base
    test_env('UrbanCausalIntersection-v0')
    # Test Extended
    test_env('UrbanCausalIntersectionExtended-v0')
    print("Verification Done.")
