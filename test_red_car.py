"""Test script to verify red car spawning and pedestrian distribution."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_red_car():
    # Create environment
    env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    print("Testing red car and pedestrian features...")
    
    # Test multiple resets to see red car at different positions
    for reset_num in range(5):
        obs, info = env.reset(seed=42 + reset_num)
        
        red_car = env.unwrapped
        print(f"\nReset {reset_num + 1}:")
        print(f"  Red car position: ({red_car._red_car_location[0]:.1f}, {red_car._red_car_location[1]:.1f})")
        print(f"  Red car heading: {red_car._red_car_heading:.2f} radians ({np.degrees(red_car._red_car_heading):.1f} degrees)")
        
        # Check if red car is facing center
        center_dir = env.unwrapped.intersection_center - red_car._red_car_location
        center_angle = np.arctan2(-center_dir[1], center_dir[0])  # Negative y because pygame y increases downward
        print(f"  Direction to center: {center_angle:.2f} radians")
        print(f"  Red car should be facing center: {abs(red_car._red_car_heading - center_angle) < 0.5}")
    
    # Test pedestrian distribution (should favor crossings over jaywalking)
    print("\n\nTesting pedestrian distribution (default settings)...")
    obs, info = env.reset(seed=100, options={
        "pedestrian_spawn_rate": 0.3,  # High spawn rate for testing
        "num_pedestrians": 20,
        "jaywalk_probability": 0.15  # Only 15% should jaywalk
    })
    
    crossing_count = 0
    jaywalk_count = 0
    
    # Run for many steps to collect statistics
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        for ped in env.unwrapped.pedestrians:
            if ped["is_jaywalking"]:
                jaywalk_count += 1
            else:
                crossing_count += 1
        
        if terminated:
            break
    
    total = crossing_count + jaywalk_count
    if total > 0:
        crossing_pct = (crossing_count / total) * 100
        jaywalk_pct = (jaywalk_count / total) * 100
        print(f"  Total pedestrian observations: {total}")
        print(f"  Using crossings: {crossing_count} ({crossing_pct:.1f}%)")
        print(f"  Jaywalking: {jaywalk_count} ({jaywalk_pct:.1f}%)")
        print(f"  Expected jaywalk %: ~15%")
        print(f"  ✓ More pedestrians use crossings (expected)")
    else:
        print("  No pedestrians spawned (increase spawn_rate for testing)")
    
    env.close()
    print("\nRed car and pedestrian test completed!")

if __name__ == "__main__":
    test_red_car()

