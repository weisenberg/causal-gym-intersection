import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from gymnasium.wrappers import RecordVideo

pygame.init()
pygame.font.init()

# Ensure env is registered
import gym_causal_intersection
import shutil
from datetime import datetime

def archive_old_runs(agent_type="ppo"):
    targets = {
        "ppo": ["videos_ppo_curved", "reward_plot_ppo_curved.png", "ppo_causal_agent_curved.zip"],
        "dqn": ["videos_dqn_curved", "reward_plot_dqn_curved.png", "dqn_causal_agent_curved.zip", "dqn_replay_buffer_curved.pkl"]
    }
    
    found = [f for f in targets[agent_type] if os.path.exists(f)]
    if not found: return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join("run_archive", f"{agent_type}_{ts}")
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

class VizCallback(BaseCallback):
    def __init__(self, viz_freq=100):
        super().__init__()
        self.viz_freq = viz_freq
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
        #             # plt.ion() # Interactive mode -> BLOCKS IN BACKGROUND
        #             # self.fig, self.ax = plt.subplots()
        #             # self.im = self.ax.imshow(frame)
        #             # plt.axis('off')
        #             pass
        #         else:
        #             # self.im.set_data(frame)
        #             pass
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
                     window = 50
                     if len(self.all_rewards) >= window:
                         ma = np.convolve(self.all_rewards, np.ones(window)/window, mode='valid')
                         plt.plot(range(window-1, len(self.all_rewards)), ma, color='red', label=f'{window}-Ep Avg')
                         
                     plt.title("PPO Episode Rewards (Curved Road)")
                     plt.xlabel("Episode")
                     plt.ylabel("Reward")
                     plt.legend()
                     plt.grid(True)
                     plt.savefig("reward_plot_ppo_curved.png")
                     plt.close()
                
                # Success Threshold (Match DQN)
                if r > 800.0:
                    print(f"Goal Reached! Reward {r:.2f} > 800.0. Stopping Training.")
                    return False
                    
            self.episode_count += 1
            
        return True

    # ... (Wrappers and imports remain similar)
    
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
    import os
    # Archive first!
    archive_old_runs("ppo")
    
    print(f"DEBUG: CWD = {os.getcwd()}")
    print(f"DEBUG: Video folder absolute path = {os.path.abspath('videos_ppo_curved')}")
    
    # 1. Create Environment
    env = gym.make('SimpleCausalIntersection-v0', render_mode='rgb_array')
    
    # 2. Add Wrappers
    env = OverlayWrapper(env) # Adds text to render()
    env = Monitor(env) # Tracks stats for SB3
    
    # 3. Add Video Recorder
    # Trigger: Record every 30th episode
    video_folder = 'videos_ppo_curved'
    def video_trigger(episode_id):
        return episode_id % 30 == 0
        
    env = RecordVideo(
        env, 
        video_folder=video_folder, 
        episode_trigger=video_trigger,
        name_prefix="ppo_agent_curved"
    )
    
    # 4. Initialize Agent
    # Use linear decay schedule (3e-4 -> 0.0)
    # New schedule takes ONE arg (initial_value)
    lr_schedule = linear_schedule(3e-4)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=lr_schedule,
        ent_coef=0.0, # Force entropy to 0.0 (User Request)
        clip_range=0.2, # Explicitly set clip range
        max_grad_norm=0.5, # Keep PPO clip at 0.5
        batch_size=64,
        n_steps=2048,
        gamma=0.99
    )
    
    print("Starting PPO training (Curved Road)...")
    print(f"Videos will be saved to ./{video_folder} every 30 episodes")
    print("Visualization window will appear every 100 episodes.")
    
    # 5. Train
    steps = 600000 # Increased Duration
    viz_callback = VizCallback(viz_freq=100)
    
    # Pass only viz_callback (Entropy is fixed 0.0)
    model.learn(total_timesteps=steps, callback=viz_callback)
    
    # 6. Save
    model.save("ppo_simple_causal_curved")
    print("Model saved to ppo_simple_causal_curved.zip")
    
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
            
        plt.title("PPO Episode Rewards (Curved Road)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("reward_plot_ppo_curved.png")
        print("Reward plot saved to reward_plot_ppo_curved.png")
    
    env.close()
    if plt.get_fignums():
        plt.close('all')

if __name__ == '__main__':
    main()
