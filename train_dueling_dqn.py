import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import pygame
import shutil
from datetime import datetime
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

pygame.init()
pygame.font.init()

# Ensure env is registered
import gym_causal_intersection

# --- DUELING DQN ARCHITECTURE ---

class DuelingQNetwork(nn.Module):
    """
    Dueling Q Network.
    Splits the last layer into two streams: Value and Advantage.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """
    def __init__(self, observation_space, action_space, features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__()
        
        # Features Extractor (The body)
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        
        action_dim = action_space.n
        
        # Define Network Architecture
        # Let's assume standard [256, 256] body if net_arch is None
        if net_arch is None:
            net_arch = [256, 256]
            
        # Common body (optional, but typical in SB3 is to put it in features or just rely on streams?)
        # SB3 "net_arch" normally defines the shared layers.
        # But for Dueling, we usually split AFTER the shared layers.
        # Let's say we have a shared body first.
        
        # To keep it compatible with SB3 logic, let's treat 'features_extractor' as the shared body if it's an MLP.
        # But usually 'features_extractor' is strictly the Flatten/CNN part.
        
        # Let's build a shared hidden layer sequence
        self.shared_net = nn.Sequential()
        input_dim = features_dim
        for i, layer_size in enumerate(net_arch):
            self.shared_net.add_module(f"fc_{i}", nn.Linear(input_dim, layer_size))
            self.shared_net.add_module(f"act_{i}", activation_fn())
            input_dim = layer_size
            
        # Now split into Value and Advantage Streams
        # Value Stream: Hidden -> 1
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation_fn(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream: Hidden -> N_Actions
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation_fn(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs):
        # Extract features
        features = self.features_extractor(obs)
        
        # Shared body
        shared_out = self.shared_net(features)
        
        # Heads
        values = self.value_stream(shared_out)
        advantages = self.advantage_stream(shared_out)
        
        # Combine: Q = V + (A - mean(A))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_vals
        
    def set_training_mode(self, mode: bool) -> None:
        """
        Sets the network to training mode or evaluation mode.
        """
        self.train(mode)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.
        
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic policy
        :return: Taken action from the policy
        """
        q_values = self(observation)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        # Override to use DuelingQNetwork
        # Robustly construct arguments since self.q_net_kwargs/net_args can be unreliable
        
        net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": self.normalize_images
        }
        
        # This Helper (from BasePolicy) adds 'features_extractor' and 'features_dim' to the dict
        net_args = self._update_features_extractor(net_args, features_extractor=None)
        
        return DuelingQNetwork(**net_args).to(self.device)


# --- HELPERS (Same as DQN Script) ---

def archive_old_runs(agent_type="dueling_dqn"):
    targets = {
        "dqn": ["videos_dqn_curved", "reward_plot_dqn_curved.png", "dqn_simple_causal_curved.zip"], # Legacy
        "dueling_dqn": ["videos_dueling_dqn", "reward_plot_dueling_dqn.png", "dueling_dqn_causal.zip"]
    }
    
    if agent_type not in targets: return
    
    found = [f for f in targets[agent_type] if os.path.exists(f)]
    if not found: return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join("run_archive", "{}_{}".format(agent_type, ts))
    os.makedirs(archive_dir, exist_ok=True)
    
    print("Archiving to {}...".format(archive_dir))
    for f in found:
        try: shutil.move(f, os.path.join(archive_dir, f))
        except Exception as e: print("Error archiving {}: {}".format(f, e))

class OverlayWrapper(gym.Wrapper):
    """
    Wrapper to overlay text information on the rendered frame using PIL.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {}
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        self.episode_reward += reward
        self.last_reward = reward
        self.last_info = info
        
        if terminated or truncated:
            self.episode_count += 1
            pass
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.current_step = 0
        return self.env.reset(**kwargs)

    def render(self):
        # Get base frame (numpy array)
        frame = self.env.render()
        
        if frame is None:
            return None
            
        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            return frame
            
        draw = ImageDraw.Draw(image)
        
        # Define text properties
        try:
            font = ImageFont.truetype("Arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        # Info to display
        left_lines = [
            f"Ep: {self.episode_count}",
            f"Step: {self.current_step}",
            f"R: {self.episode_reward:.1f}",
            f"Last: {self.last_reward:.1f}"
        ]
        
        right_lines = []
        if "causal_vars" in self.last_info:
            cv = self.last_info["causal_vars"]
            for k, v in cv.items():
                if isinstance(v, (int, float)):
                    right_lines.append(f"{k}: {v:.1f}")
                else:
                    right_lines.append(f"{k}: {v}")

        # Draw left text
        y0, dy = 10, 15
        x_left = 10
        for i, line in enumerate(left_lines):
            y = y0 + i * dy
            draw.text((x_left+1, y+1), line, font=font, fill=(0,0,0))
            draw.text((x_left, y), line, font=font, fill=(255,255,255))
            
        # Draw right text
        img_width = image.width
        for i, line in enumerate(right_lines):
            y = y0 + i * dy
            text_width = font.getlength(line)
            x_right = img_width - text_width - 10
            draw.text((x_right+1, y+1), line, font=font, fill=(0,0,0))
            draw.text((x_right, y), line, font=font, fill=(255,255,255))
            
        return np.array(image)

class VizCallback(BaseCallback):
    def __init__(self, viz_freq=100):
        super().__init__()
        self.viz_freq = viz_freq
        self.episode_count = 0
        self.all_rewards = []
    
    def _on_step(self):
        # Check for done
        if self.locals['dones'][0]:
            ep_info = self.locals['infos'][0].get('episode')
            if ep_info:
                r = ep_info['r']
                self.all_rewards.append(r)
                print(f"Episode {self.episode_count}: Reward = {r:.2f}")
                
                # Save plot periodically (every 10 episodes)
                if self.episode_count % 10 == 0:
                     plt.figure(figsize=(10, 5))
                     plt.plot(self.all_rewards, alpha=0.3, label='Raw')
                     # Rolling Mean
                     window = 50
                     if len(self.all_rewards) >= window:
                         ma = np.convolve(self.all_rewards, np.ones(window)/window, mode='valid')
                         plt.plot(range(window-1, len(self.all_rewards)), ma, color='red', label=f'{window}-Ep Avg')
                         
                     plt.title("Dueling DQN Rewards")
                     plt.xlabel("Episode")
                     plt.ylabel("Reward")
                     plt.legend()
                     plt.grid(True)
                     plt.savefig("reward_plot_dueling_dqn.png")
                     plt.close()
                
                # Success Threshold
                if r > 800.0:
                    print(f"Goal Reached! Reward {r:.2f} > 800.0. Stopping Training.")
                    return False # Stop training
                    
            self.episode_count += 1
            
        return True

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    # Archive first
    archive_old_runs("dueling_dqn")

    # 1. Create Environment
    env = gym.make('SimpleCausalIntersection-v0', render_mode='rgb_array')
    env = OverlayWrapper(env)
    env = Monitor(env) 
    
    # 2. Add Video Recorder
    video_folder = 'videos_dueling_dqn'
    def video_trigger(episode_id):
        return episode_id % 30 == 0
        
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=video_trigger,
        name_prefix="dueling_dqn"
    )
    
    # 3. Initialize DUELING Agent
    # We use our Custom DuelingDQNPolicy
    model = DQN(
        DuelingDQNPolicy, # Custom Policy
        env,
        learning_rate=linear_schedule(1e-4),
        buffer_size=200000,
        learning_starts=5000,
        batch_size=128,
        tau=0.05,
        gamma=0.98,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5000,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        max_grad_norm=1.0, # Enable Gradient Clipping
        verbose=1,
        policy_kwargs={"net_arch": [256, 256]} # Passed to our Custom Policy
    )
    
    print("Starting Dueling DQN training...")
    print(f"Videos: ./{video_folder}")
    
    viz_callback = VizCallback(viz_freq=100)
    model.learn(total_timesteps=600000, callback=viz_callback)
    
    model.save("dueling_dqn_causal")
    print("Model saved to dueling_dqn_causal.zip")
    
    # Final Plot
    rewards = viz_callback.all_rewards
    if rewards:
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label='Raw')
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), ma, color='red', label=f'{window}-Ep Avg')
        plt.title("Dueling DQN Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("reward_plot_dueling_dqn.png")
    
    env.close()

if __name__ == '__main__':
    main()
