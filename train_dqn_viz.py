import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from gymnasium.wrappers import RecordVideo
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

# --- DUELING DQN ARCHITECTURE ---
class DuelingQNetwork(nn.Module):
    def __init__(self, observation_space, action_space, features_extractor, features_dim,
                 net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        action_dim = action_space.n
        
        if net_arch is None: net_arch = [256, 256]
            
        self.shared_net = nn.Sequential()
        input_dim = features_dim
        for i, layer_size in enumerate(net_arch):
            self.shared_net.add_module(f"fc_{i}", nn.Linear(input_dim, layer_size))
            self.shared_net.add_module(f"act_{i}", activation_fn())
            input_dim = layer_size
            
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation_fn(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            activation_fn(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, obs):
        features = self.features_extractor(obs)
        shared_out = self.shared_net(features)
        values = self.value_stream(shared_out)
        advantages = self.advantage_stream(shared_out)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": self.normalize_images
        }
        net_args = self._update_features_extractor(net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)

pygame.init()
pygame.font.init()

# Ensure env is registered
import gym_causal_intersection
import shutil
from datetime import datetime

def archive_old_runs(agent_type="dqn"):
    log_root = "/Users/ali/Desktop/masterarbeit/playground/logs"
    targets = {
        "ppo": ["videos_ppo_curved", "reward_plot_ppo_curved.png", "ppo_causal_agent_curved.zip"],
        "dqn": ["videos_dqn_curved", "reward_plot_dqn_curved.png", "dqn_causal_agent_curved.zip", "dqn_replay_buffer_curved.pkl"]
    }
    
    found = [f for f in targets[agent_type] if os.path.exists(f)]
    if not found: return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(log_root, "run_archive", f"{agent_type}_{ts}")
    os.makedirs(archive_dir, exist_ok=True)
    
    print(f"Archiving to {archive_dir}...")
    for f in found:
        try: shutil.move(f, os.path.join(archive_dir, f))
        except Exception as e: print(f"Error archiving {f}: {e}")

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
        # Load default font or specific one if available
        try:
            # Try to load a reasonable font
            font = ImageFont.truetype("Arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
            
        # Info to display
        left_lines = [
            f"Episode: {self.episode_count}",
            f"Step: {self.current_step}",
            f"Ep Reward: {self.episode_reward:.2f}",
            f"Last Reward: {self.last_reward:.2f}"
        ]
        
        right_lines = []
        # Add causal vars if available
        if "causal_vars" in self.last_info:
            cv = self.last_info["causal_vars"]
            for k, v in cv.items():
                if isinstance(v, (int, float)):
                    right_lines.append(f"{k}: {v:.2f}")
                else:
                    right_lines.append(f"{k}: {v}")

        # Draw left text
        y0, dy = 10, 15
        x_left = 10
        for i, line in enumerate(left_lines):
            y = y0 + i * dy
            # Draw shadow
            draw.text((x_left+1, y+1), line, font=font, fill=(0,0,0))
            # Draw text
            draw.text((x_left, y), line, font=font, fill=(255,255,255))
            
        # Draw right text (Additional Info)
        # We need image width
        img_width = image.width
        for i, line in enumerate(right_lines):
            y = y0 + i * dy
            # Calculate text width
            text_width = font.getlength(line)
            x_right = img_width - text_width - 10
            
            # Draw shadow
            draw.text((x_right+1, y+1), line, font=font, fill=(0,0,0))
            # Draw text
            draw.text((x_right, y), line, font=font, fill=(255,255,255))
            
        # Convert back to numpy
        return np.array(image)

class DiscreteWrapper(gym.ActionWrapper):
    """
    Wrapper to convert Discrete actions to Continuous [accel, steer] for DQN validation.
    """
    def __init__(self, env):
        super().__init__(env)
        # 0: Hard Brake, 1: Cruise, 2: Accel, 3: Left, 4: Right, 5: Hard Left, 6: Hard Right
        self.action_space = gym.spaces.Discrete(7)
        
    def action(self, action):
        # Map Discrete(0..6) to Continuous [accel, steer]
        # Handle numpy scalars
        if hasattr(action, 'item'):
            action = action.item()
            
        if action == 0:   # Hard Brake
            return np.array([-1.0, 0.0], dtype=np.float32)
        elif action == 1: # Cruise
            return np.array([0.0, 0.0], dtype=np.float32)
        elif action == 2: # Accelerate
            return np.array([1.0, 0.0], dtype=np.float32)
        elif action == 3: # Steer Left
            # User Reqs: Acc=0.5, Steer=-0.5
            return np.array([0.5, -0.5], dtype=np.float32)
        elif action == 4: # Steer Right
            # User Reqs: Acc=0.5, Steer=0.5
            return np.array([0.5, 0.5], dtype=np.float32)
        elif action == 5: # Hard Left
            # User Reqs: Acc=0.2, Steer=-1.0
            return np.array([0.2, -1.0], dtype=np.float32)
        elif action == 6: # Hard Right
            # User Reqs: Acc=0.2, Steer=1.0
            return np.array([0.2, 1.0], dtype=np.float32)
        # Default safety
        return np.array([0.0, 0.0], dtype=np.float32)

class VizCallback(BaseCallback):
    def __init__(self, viz_freq=100, save_path=".", plot_name="reward_plot_dqn_curved.png"):
        super().__init__()
        self.viz_freq = viz_freq
        self.save_path = save_path
        self.plot_name = plot_name
        self.episode_count = 0
        self.fig, self.ax = None, None
        self.all_rewards = []
    
    def _on_step(self):
        # Render if needed
        # Render if needed
        # if self.episode_count % self.viz_freq == 0:
        #     frame = self.training_env.envs[0].render()
        #     if frame is not None:
        #         if self.fig is None:
        #             pass # plt.ion()
        #             # self.fig, self.ax = plt.subplots()
        #             # self.im = self.ax.imshow(frame)
        #             # plt.axis('off')
        #         else:
        #             pass # self.im.set_data(frame)
        #         
        #         # plt.draw()
        #         # plt.pause(0.001)
        pass
        
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
                     if len(self.all_rewards) > 0:
                         window = min(50, len(self.all_rewards))
                         if window > 1:
                             ma = np.convolve(self.all_rewards, np.ones(window)/window, mode='valid')
                             plt.plot(range(window-1, len(self.all_rewards)), ma, color='red', label=f'{window}-Ep Avg')
                         
                     plt.title("DQN Episode Rewards (Curved Road)")
                     plt.xlabel("Episode")
                     plt.ylabel("Reward")
                     plt.legend()
                     plt.grid(True)
                     plt.savefig(os.path.join(self.save_path, self.plot_name))
                     plt.close()
                
                # Success Threshold Check (User Req: ~50s driving)
                # Est: 1500 steps * ~0.6 reward/step = 900.
                # Threshold: 800.0
                # Success Threshold Check (User Req: ~50s driving)
                # Est: 1500 steps * ~0.6 reward/step = 900.
                # Threshold: 800.0
                # removed per user request for max potential
                # if r > 800.0:
                #     print(f"Goal Reached! Reward {r:.2f} > 800.0. Stopping Training.")
                #     return False # Stop training
                    
            self.episode_count += 1
            
        return True

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining decreases from 1.0 (start) to 0.0 (end)
        return progress_remaining * initial_value
    return func

def main():
    # Archive first!
    archive_old_runs("dqn")

    # Define Log Directory
    log_root = "/Users/ali/Desktop/masterarbeit/playground/logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dqn_viz_{timestamp}"
    run_dir = os.path.join(log_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training run directory: {run_dir}")

    # 1. Create Environment
    env = gym.make('SimpleCausalIntersection-v0', render_mode='rgb_array')
    
    # 2. Add Wrappers
    from gym_causal_intersection.wrappers.safety_wrapper import SafetyWrapper
    
    # Order: Agent -> DiscreteWrapper -> SafetyWrapper -> Env
    # SafetyWrapper sees Continuous Actions produced by DiscreteWrapper
    
    env = SafetyWrapper(env) 
    env = DiscreteWrapper(env)
    
    env = OverlayWrapper(env) # Adds text to render()
    env = Monitor(env, filename=os.path.join(run_dir, "monitor")) # Tracks stats for SB3
    
    # 3. Add Video Recorder
    # Trigger: Record every 30th episode
    video_folder = os.path.join(run_dir, 'videos_dqn_curved')
    def video_trigger(episode_id):
        return episode_id % 30 == 0
        
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=video_trigger,
        name_prefix="dqn_agent_curved"
    )
    
    # 4. Initialize Agent
    # Create the agent
    model = DQN(
        DuelingDQNPolicy, 
        env, 
        verbose=1, 
        learning_rate=1e-4, 
        buffer_size=100000, 
        learning_starts=1000, 
        batch_size=64, 
        tau=0.05, 
        gamma=0.99, # Focus on future
        train_freq=4, 
        gradient_steps=1, # User Req
        target_update_interval=1000,
        exploration_fraction=0.4, # User Req: 0.4
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01, # Decay to 1%
        max_grad_norm=1.0, # Cap gradients
        policy_kwargs=dict(net_arch=[256, 256]), # User Req: Dueling is implicit in Policy class
        tensorboard_log=os.path.join(run_dir, "tensorboard")
    )
    
    # Callbacks
    viz_callback = VizCallback(viz_freq=30, save_path=run_dir, plot_name="reward_plot_dqn.png")
    
    # Eval Callback (Best Model)
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
    
    # Needs a separate Eval Env with same wrappers!
    eval_env = gym.make('SimpleCausalIntersection-v0', render_mode=None)
    eval_env = SafetyWrapper(eval_env)
    eval_env = DiscreteWrapper(eval_env)
    eval_env = Monitor(eval_env)

    # Save best model to ./logs/best_model
    # Check every 10,000 steps
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, 'best_model_dqn'),
        log_path=os.path.join(run_dir, 'results_dqn'),
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_train_callback
    )
    
    # Chain callbacks
    # Chain callbacks
    # Removed StopTrainingOnRewardThreshold
    callbacks = [viz_callback, eval_callback]
    
    print("Starting DQN training (Curved Road)...")
    print(f"Videos will be saved to {video_folder} every 30 episodes")
    print(f"Best model will be saved to {os.path.join(run_dir, 'best_model_dqn')}")
    
    # Increased total steps to allow convergence
    model.learn(total_timesteps=600000, callback=callbacks)
    
    # 6. Save
    model.save(os.path.join(run_dir, "dqn_simple_causal_curved"))
    print(f"Model saved to {os.path.join(run_dir, 'dqn_simple_causal_curved.zip')}")
    
    # 7. Plot Rewards
    rewards = viz_callback.all_rewards
    if rewards:
        final_reward = rewards[-1]
        print(f"Final Reward: {final_reward}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.3, label='Raw')
        
        # Rolling Mean
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), ma, color='red', label=f'{window}-Ep Avg')
            
        plt.title("DQN Episode Rewards (Curved Road)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, "reward_plot_dqn_curved.png"))
        print(f"Reward plot saved to {os.path.join(run_dir, 'reward_plot_dqn_curved.png')}")
    
    env.close()
    if plt.get_fignums():
        plt.close('all')

if __name__ == '__main__':
    main()
