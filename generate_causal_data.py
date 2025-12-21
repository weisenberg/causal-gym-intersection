import gymnasium as gym
import gym_causal_intersection
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

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
        # self.model._last_obs: The observation BEFORE the step (numpy array)
        # self.locals['actions']: The action taken
        # self.locals['rewards']: The reward received
        # self.locals['dones']: Whether the episode ended
        # self.locals['infos']: Info dictionary
        
        obs = self.model._last_obs
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']
        
        # Handle vectorized environments (SB3 uses VecEnv by default)
        # We assume num_envs=1 for simplicity in this script
        
        # Loop through environments (usually just 1)
        for i in range(len(obs)):
            row = {}
            
            # Flatten Observation (55-dim)
            # obs[i] is the observation for env i
            curr_obs = obs[i]
            # Flatten Observation (55-dim) with meaningful names
            # obs[i] is the observation for env i
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
            # Features: rel_x, rel_y, rel_vx, rel_vy
            base_idx = 21
            for c in range(5):
                row[f"car{c+1}_rel_x"] = curr_obs[base_idx + c*4 + 0]
                row[f"car{c+1}_rel_y"] = curr_obs[base_idx + c*4 + 1]
                row[f"car{c+1}_rel_vx"] = curr_obs[base_idx + c*4 + 2]
                row[f"car{c+1}_rel_vy"] = curr_obs[base_idx + c*4 + 3]
                
            # 4. Nearest Pedestrians (5 peds * 2 features = 10)
            # Features: rel_x, rel_y
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
            # infos[i] is the info dict for env i
            cv = infos[i].get("causal_vars", {})
            for k, v in cv.items():
                if isinstance(v, str):
                    # Map strings to simple integers for causal discovery
                    if k == "traffic_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    elif k == "pedestrian_density":
                        map_d = {"low": 0, "medium": 1, "high": 2}
                        row[k] = map_d.get(v, -1)
                    elif k == "npc_color":
                        # Simple hash for color
                        row[k] = hash(v) % 100
                    elif k == "npc_size":
                         map_s = {"random": 0, "small": 1, "medium": 2, "large": 3}
                         row[k] = map_s.get(v, 0)
                    else:
                        row[k] = -1
                else:
                    row[k] = v
            
            # Add Action
            # actions[i] is [accel, steer]
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

def generate_rl_data(total_timesteps=50000, output_file="causal_discovery_dataset.csv"):
    # Create environment
    # We use the standard gym.make, SB3 will wrap it in a DummyVecEnv automatically
    env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    # Initialize PPO Agent
    # We use MlpPolicy because inputs are kinematic (vector), not images
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Initialize Callback
    callback = DataLoggingCallback()
    
    print(f"Starting RL training for {total_timesteps} steps...")
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the logged data
    callback.save_data(output_file)
    
    env.close()

if __name__ == "__main__":
    generate_rl_data()
