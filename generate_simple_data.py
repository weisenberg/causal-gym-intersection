import gymnasium as gym
import gym_causal_intersection
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

class DataLoggingCallback(BaseCallback):
    """
    Callback for logging data during training for SimpleCausalIntersection-v0.
    """
    def __init__(self, verbose=0):
        super(DataLoggingCallback, self).__init__(verbose)
        self.data_rows = []
        self.episode_count = 0
        self.last_dones = [False]

    def _on_step(self) -> bool:
        obs = self.model._last_obs
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        # Access the unwrapped environment to get raw state
        raw_env = self.training_env.envs[0].unwrapped
        
        # Discretization Helpers (Bins)
        # Discretization Helpers (Bins)
        # Position: 600 bins (Pixel level)
        # Old Env: 0-600. New Env: -1000 to +1000 (effectively).
        # We'll map -1000..1000 to 0..600 bins
        def _bin_pos(val, min_val=-1000.0, max_val=1000.0, bins=600):
            norm = (val - min_val) / (max_val - min_val)
            return int(np.clip(norm * bins, 0, bins-1))
            
        # Angle: 50 bins
        def _bin_angle(heading, bins=50):
            # Normalize to 0-1 then bin
            norm = (heading + np.pi) / (2 * np.pi)
            return int(np.clip(norm * bins, 0, bins-1))
            
        # Speed: 50 bins (-5 to 5 maps to magnitude 0-5 -> bins)
        # Actually obs has velocity commands 0-1. 
        # But here we look at raw velocity magnitude. Max speed approx 5.
        def _bin_speed(vel, max_speed=5.0, bins=50):
            s = np.linalg.norm(vel)
            return int(np.clip(s / (max_speed/bins), 0, bins-1))

        # Loop through environments (usually just 1)
        for i in range(len(obs)):
            if self.last_dones[i]:
                self.episode_count += 1
            
            row = {}
            row["episode"] = self.episode_count
            row["step"] = raw_env.step_count
            
            # Simple Env Observation (Normalized Continuous 0-1)
            # We log these as floats for reference of what the agent SAW
            curr_obs = obs[i]
            # State vector is now large (~200 dims). X,Y are at 0,1.
            row["obs_agent_x_normalized"] = float(curr_obs[0])
            row["obs_agent_y_normalized"] = float(curr_obs[1])
            
            # Explicit Discrete Variables (Requested by User)
            row["agent_x_position"] = _bin_pos(raw_env._agent_location[0])
            row["agent_y_position"] = _bin_pos(raw_env._agent_location[1])
            row["agent_speed_bin"] = _bin_speed(raw_env._agent_velocity)
            row["agent_angle_bin"] = _bin_angle(raw_env._agent_heading)
            
            # NPC Cars (Max 2)
            npc_cars = raw_env.npc_cars
            for idx in range(2): 
                prefix = f"npc_{idx+1}"
                if idx < len(npc_cars):
                    car = npc_cars[idx]
                    row[f"{prefix}_present"] = 1
                    row[f"{prefix}_x_position"] = _bin_pos(car["pos"][0])
                    row[f"{prefix}_y_position"] = _bin_pos(car["pos"][1])
                    row[f"{prefix}_angle_bin"] = _bin_angle(car["heading"])
                    row[f"{prefix}_speed_bin"] = _bin_speed(car["velocity"])
                    row[f"{prefix}_stopping_dist"] = int(car.get("stopping_distance", 40.0))
                    row[f"{prefix}_color_r"] = int(car.get("color", (0,0,255))[0])
                    row[f"{prefix}_color_g"] = int(car.get("color", (0,0,255))[1])
                    row[f"{prefix}_color_b"] = int(car.get("color", (0,0,255))[2])
                else:
                    row[f"{prefix}_present"] = 0
                    row[f"{prefix}_x_position"] = 0
                    row[f"{prefix}_y_position"] = 0
                    row[f"{prefix}_angle_bin"] = 0
                    row[f"{prefix}_speed_bin"] = 0
                    row[f"{prefix}_stopping_dist"] = 0
                    row[f"{prefix}_color_r"] = 0
                    row[f"{prefix}_color_g"] = 0
                    row[f"{prefix}_color_b"] = 0

            # Pedestrians (Max 2)
            pedestrians = raw_env.pedestrians
            for idx in range(2):
                prefix = f"ped_{idx+1}"
                if idx < len(pedestrians):
                    ped = pedestrians[idx]
                    row[f"{prefix}_present"] = 1
                    row[f"{prefix}_x_position"] = _bin_pos(ped["pos"][0])
                    row[f"{prefix}_y_position"] = _bin_pos(ped["pos"][1])
                    row[f"{prefix}_jaywalking"] = 1 if ped["is_jaywalking"] else 0
                else:
                    row[f"{prefix}_present"] = 0
                    row[f"{prefix}_x_position"] = 0
                    row[f"{prefix}_y_position"] = 0
                    row[f"{prefix}_jaywalking"] = 0
            
            # Add Hidden Confounders (Mapped to Ints)
            cv = infos[i].get("causal_vars", {})
            for k, v in cv.items():
                if isinstance(v, str):
                    if k == "traffic_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    elif k == "pedestrian_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    else:
                        row[k] = -1
                else:
                    row[k] = float(v) 
            
            # Add Action
            row["action"] = int(actions[i])
            
            # Add Reward
            row["reward"] = float(rewards[i]) 
            row["done"] = 1 if dones[i] else 0
            
            self.data_rows.append(row)
            self.last_dones[i] = dones[i]
            
        return True

    def save_data(self, output_file):
        df = pd.DataFrame(self.data_rows)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")

def generate_simple_data(total_timesteps=1000000, output_file="simple_env_data.csv"):
    # Create environment
    env = gym.make('SimpleCausalIntersection-v0', render_mode=None)
    
    # Initialize PPO Agent
    # MlpPolicy works for MultiDiscrete inputs too
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Initialize Callback
    callback = DataLoggingCallback()
    
    print(f"Starting RL training for {total_timesteps} steps on Simple Env...")
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the logged data
    callback.save_data(output_file)
    
    env.close()

if __name__ == "__main__":
    generate_simple_data()
