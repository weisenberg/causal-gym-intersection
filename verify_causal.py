import gymnasium as gym
import gym_causal_intersection
import numpy as np

def verify_causal(env_name):
    print(f"\n{'='*60}")
    print(f"Verifying Causal RL Features for {env_name}")
    print(f"{'='*60}")
    
    env = gym.make(env_name, render_mode=None)
    
    # Test 1: Domain Randomization (Observational)
    print("\nTest 1: Domain Randomization (Observational)")
    print("Resetting 5 times without options...")
    for i in range(5):
        obs, info = env.reset()
        cv = info["causal_vars"]
        print(f"  Run {i+1}: Temp={cv['temperature']}, Traffic={cv['traffic_density']}, "
              f"Peds={cv['pedestrian_density']}, Impatience={cv['driver_impatience']:.2f}")
        
    # Test 2: Interventions (Do-Calculus)
    print("\nTest 2: Interventions (Do-Calculus)")
    intervention = {
        "temperature": -10,
        "traffic_density": "high",
        "pedestrian_density": "low",
        "driver_impatience": 0.95,
        "npc_color": "red",
        "npc_size": "large"
    }
    print(f"Intervening with: {intervention}")
    print("Resetting 3 times WITH options...")
    for i in range(3):
        obs, info = env.reset(options=intervention)
        cv = info["causal_vars"]
        print(f"  Run {i+1}: Temp={cv['temperature']}, Traffic={cv['traffic_density']}, "
              f"Color={cv['npc_color']}, Size={cv['npc_size']}")
        
        # Verify values match intervention
        assert cv["temperature"] == -10
        assert cv["traffic_density"] == "high"
        assert cv["pedestrian_density"] == "low"
        assert cv["driver_impatience"] == 0.95
        assert cv["npc_color"] == "red"
        assert cv["npc_size"] == "large"
        
    print(">> Intervention Test PASSED: Environment correctly forced causal variables.")
    
    # Test 3: Data Extraction (Step Info)
    print("\nTest 3: Data Extraction (Step Info)")
    obs, info = env.reset()
    print("Taking a step...")
    obs, reward, terminated, truncated, info = env.step([1.0, 0.0])
    if "causal_vars" in info:
        print(f"  Step Info contains causal_vars: {info['causal_vars']}")
        print(">> Data Extraction Test PASSED.")
    else:
        print(">> Data Extraction Test FAILED: 'causal_vars' missing from step info.")

    env.close()

if __name__ == "__main__":
    verify_causal('UrbanCausalIntersection-v0')
    verify_causal('UrbanCausalIntersectionExtended-v0')
