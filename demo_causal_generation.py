import gymnasium as gym
import gym_causal_intersection
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import time

class DataLoggingCallback(BaseCallback):
    """
    Callback for logging data during training.
    """
    def __init__(self, verbose=0):
        super(DataLoggingCallback, self).__init__(verbose)
        self.data_rows = []
        self.episode_counts = 0

    def _on_step(self) -> bool:
        # Access local variables from the model's learn method
        obs = self.model._last_obs
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        # Loop through environments (usually just 1)
        for i in range(len(obs)):
            row = {}
            
            # Flatten Observation (55-dim) with meaningful names
            curr_obs = obs[i]
            
            # 1. Agent State (5)
            row["agent_x"] = curr_obs[0]
            row["agent_y"] = curr_obs[1]
            row["agent_vx"] = curr_obs[2]
            row["agent_vy"] = curr_obs[3]
            row["agent_heading"] = curr_obs[4]
            
            # 2. Lidar (16)
            for l in range(16):
                row[f"lidar_{l}"] = curr_obs[5 + l]
                
            # 3. Nearest Cars (5 cars * 4 features = 20)
            base_idx = 21
            for c in range(5):
                row[f"car{c+1}_rel_x"] = curr_obs[base_idx + c*4 + 0]
                row[f"car{c+1}_rel_y"] = curr_obs[base_idx + c*4 + 1]
                row[f"car{c+1}_rel_vx"] = curr_obs[base_idx + c*4 + 2]
                row[f"car{c+1}_rel_vy"] = curr_obs[base_idx + c*4 + 3]
                
            # 4. Nearest Pedestrians (5 peds * 2 features = 10)
            base_idx = 41
            for p in range(5):
                row[f"ped{p+1}_rel_x"] = curr_obs[base_idx + p*2 + 0]
                row[f"ped{p+1}_rel_y"] = curr_obs[base_idx + p*2 + 1]
                
            # 5. Traffic Light (4)
            base_idx = 51
            row["tl_red"] = curr_obs[base_idx]
            row["tl_yellow"] = curr_obs[base_idx + 1]
            row["tl_green"] = curr_obs[base_idx + 2]
            row["tl_ttc"] = curr_obs[base_idx + 3]
            
            # Add Hidden Confounders (from infos)
            cv = infos[i].get("causal_vars", {})
            for k, v in cv.items():
                if isinstance(v, str):
                    if k == "traffic_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    elif k == "pedestrian_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    elif k == "npc_color":
                        row[k] = hash(v) % 100
                    elif k == "npc_size":
                         map_s = {"random": 0, "small": 1, "medium": 2, "large": 3}
                         row[k] = map_s.get(v, 0)
                    else:
                        row[k] = -1
                else:
                    row[k] = v
            
            # Add Action
            row["action_accel"] = float(actions[i][0])
            row["action_steer"] = float(actions[i][1])
            
            # Add Reward and Done
            row["reward"] = float(rewards[i])
            row["done"] = 1 if dones[i] else 0
            
            self.data_rows.append(row)
            
        return True

    def save_data(self, output_file):
        df = pd.DataFrame(self.data_rows)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")

def run_demo(total_timesteps=2000, output_file="demo_data.csv"):
    print("Starting visual demo of causal data generation...")
    print("This will open a window showing the agent training and collecting data.")
    
    # Create environment with human rendering
    env = gym.make('UrbanCausalIntersection-v0', render_mode="human")
    
    # Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Initialize Callback
    callback = DataLoggingCallback()
    
    # Train the agent (this runs the simulation)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the logged data
    callback.save_data(output_file)
    
    env.close()
    print("Demo complete!")

if __name__ == "__main__":
    run_demo()
