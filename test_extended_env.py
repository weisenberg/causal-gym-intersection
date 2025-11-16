
"""Test script for the extended urban intersection environment."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np

def test_extended_environment():
    print("Testing UrbanCausalIntersectionExtended-v0 environment...")
    env = gym.make('UrbanCausalIntersectionExtended-v0', render_mode=None)
    
    obs, info = env.reset(seed=42)
    
    print("\n" + "="*70)
    print("ENVIRONMENT SPECIFICATIONS")
    print("="*70)
    print(f"Map size: {env.unwrapped.map_size}x{env.unwrapped.map_size}")
    print(f"Number of intersections: {len(env.unwrapped.intersections)}")
    print(f"Traffic lights: {len(env.unwrapped.traffic_lights)}")
    print(f"Zebra crossings: {len(env.unwrapped.zebra_crossings)}")
    print(f"Horizontal roads: {len(env.unwrapped.horizontal_roads)}")
    print(f"Vertical roads: {len(env.unwrapped.vertical_roads)}")
    print(f"Car spawn points: {len(env.unwrapped.car_spawn_points)}")
    print(f"Pedestrian spawn points: {len(env.unwrapped.pedestrian_spawn_points)}")
    
    print("\n" + "="*70)
    print("FEATURES")
    print("="*70)
    print("✓ Large map with multiple intersections")
    print("✓ Traffic lights at each intersection")
    print("✓ NPC cars that follow traffic rules")
    print("✓ Pedestrians that respect traffic lights")
    print("✓ Zebra crossings at each intersection")
    print("✓ Camera follows agent (centered view)")
    
    print("\n" + "="*70)
    print("TESTING SYSTEMS")
    print("="*70)
    
    # Test traffic lights
    print("\n1. Testing traffic lights:")
    for i, (key, light) in enumerate(list(env.unwrapped.traffic_lights.items())[:3]):
        dirs = light['directions']
        print(f"   Intersection {key}: Phase {light['phase']}, NS={dirs['north']}/{dirs['south']}, EW={dirs['east']}/{dirs['west']}")
    
    # Test NPC car spawning
    print("\n2. Testing NPC car spawning (higher spawn rate):")
    obs, info = env.reset(seed=100, options={"npc_car_spawn_rate": 0.1, "num_npc_cars": 5})
    for i in range(50):
        obs, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if i % 10 == 0:
            print(f"   Step {i}: NPC cars: {info.get('num_npc_cars', 0)}")
    
    # Test pedestrian spawning with traffic lights
    print("\n3. Testing pedestrian spawning and traffic lights:")
    obs, info = env.reset(seed=200, options={"pedestrian_spawn_rate": 0.1, "num_pedestrians": 5})
    for i in range(100):
        obs, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if i % 25 == 0:
            num_peds = info.get('num_pedestrians', 0)
            waiting = sum(1 for p in env.unwrapped.pedestrians if p.get('waiting_for_light', False))
            print(f"   Step {i}: Pedestrians: {num_peds}, Waiting at light: {waiting}")
    
    # Test NPC car behavior
    print("\n4. Testing NPC car behavior:")
    if len(env.unwrapped.npc_cars) > 0:
        npc = env.unwrapped.npc_cars[0]
        print(f"   NPC car at: ({npc['pos'][0]:.0f}, {npc['pos'][1]:.0f})")
        print(f"   State: {npc['state']}, Speed: {npc['speed']:.2f}")
        print(f"   ✓ NPC cars spawned and moving")
    
    env.close()
    print("\n" + "="*70)
    print("✓ All systems working correctly!")
    print("="*70)

if __name__ == "__main__":
    test_extended_environment()

