import gymnasium as gym
import gym_causal_intersection
import numpy as np

def verify_env(env_name):
    print(f"\nVerifying {env_name}...")
    env = gym.make(env_name, render_mode=None)
    
    temps_seen = set()
    physics_data = []
    
    for i in range(20):
        obs, info = env.reset()
        temp = env.unwrapped.context["temperature"]
        roughness = env.unwrapped.context["roughness"]
        friction = env.unwrapped.context["friction"]
        
        # Access internal physics modifiers if possible
        npc_speed_factor = getattr(env.unwrapped, "npc_speed_factor", "N/A")
        npc_brake_mod = getattr(env.unwrapped, "npc_brake_mod", "N/A")
        
        temps_seen.add(temp)
        physics_data.append({
            "temp": temp,
            "roughness": roughness,
            "friction": friction,
            "speed_factor": npc_speed_factor,
            "brake_mod": npc_brake_mod
        })
    
    # Sort by temp and print summary
    physics_data.sort(key=lambda x: x["temp"])
    
    print(f"Unique temperatures seen: {sorted(list(temps_seen))}")
    
    # Check trends
    print("\nPhysics Parameters by Temperature:")
    unique_entries = {x["temp"]: x for x in physics_data}.values()
    for data in sorted(unique_entries, key=lambda x: x["temp"]):
        print(f"Temp: {data['temp']} C | Roughness: {data['roughness']:.2f} | Friction: {data['friction']:.2f} | NPC Speed Factor: {data['speed_factor']:.2f} | Brake Mod: {data['brake_mod']:.3f}")

    env.close()

if __name__ == "__main__":
    verify_env('UrbanCausalIntersection-v0')
    verify_env('UrbanCausalIntersectionExtended-v0')
