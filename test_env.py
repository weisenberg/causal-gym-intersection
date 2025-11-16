"""Simple test script to verify the UrbanCausalIntersection-v0 environment works."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_environment():
    # Create the environment
    env = gym.make('UrbanCausalIntersection-v0', render_mode='human')
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Default context: {env.unwrapped.context}")
    
    # Test reset with default context
    obs, info = env.reset()
    print(f"\nReset with default context:")
    print(f"Initial observation: {obs}")
    print(f"Info: {info}")
    
    # Test reset with custom context
    obs, info = env.reset(options={"friction": 0.95, "traffic_light_duration": 20})
    print(f"\nReset with custom context:")
    print(f"Updated context: {env.unwrapped.context}")
    print(f"Initial observation: {obs}")
    
    # Test a few steps
    print("\nTesting actions...")
    # Accelerate forward, steer left, accelerate, steer right, brake (negative accel)
    actions_to_test = [
        np.array([0.8, 0.0], dtype=np.float32),
        np.array([0.8, -0.5], dtype=np.float32),
        np.array([0.8, 0.0], dtype=np.float32),
        np.array([0.8, 0.5], dtype=np.float32),
        np.array([-0.8, 0.0], dtype=np.float32),
    ]
    for i, action in enumerate(actions_to_test):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}, Action: {action.tolist()}, Reward: {reward:.2f}, Terminated: {terminated}")
        # Kinematic mode: 55-dim vector; print first 6 for sanity
        print(f"  Observation (first 6): {obs[:6]}")
        if terminated:
            print("  Episode terminated!")
            break
    
    # Test off-screen termination
    print("\nTesting off-screen termination...")
    obs, info = env.reset()
    # Move agent off-screen by setting position directly (for testing)
    env.unwrapped._agent_location = np.array([700.0, 300.0])  # Off-screen x position
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))  # Idle action
    print(f"Off-screen test - Reward: {reward:.2f}, Terminated: {terminated}")
    assert terminated == True, "Should terminate when off-screen"
    assert reward >= 10.0, "Should give at least +10.0 reward for going off-screen without collision"
    print("Off-screen termination test passed!")
    
    env.close()
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_environment()

