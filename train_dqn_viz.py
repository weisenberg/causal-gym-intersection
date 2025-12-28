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

pygame.init()
pygame.font.init()

# Ensure env is registered
import gym_causal_intersection
import shutil
from datetime import datetime

def archive_old_runs(agent_type="dqn"):
    targets = {
        "ppo": ["videos_ppo_curved", "reward_plot_ppo_curved.png", "ppo_causal_agent_curved.zip"],
        "dqn": ["videos_dqn_curved", "reward_plot_dqn_curved.png", "dqn_causal_agent_curved.zip", "dqn_replay_buffer_curved.pkl"]
    }
    
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
                     plt.plot(self.all_rewards)
                     plt.title("DQN Episode Rewards (Curved Road)")
                     plt.xlabel("Episode")
                     plt.ylabel("Reward")
                     plt.grid(True)
                     plt.savefig("reward_plot_dqn_curved.png")
                     plt.close()
                
                # Success Threshold Check (User Req: ~50s driving)
                # Est: 1500 steps * ~0.6 reward/step = 900.
                # Threshold: 800.0
                if r > 800.0:
                    print(f"Goal Reached! Reward {r:.2f} > 800.0. Stopping Training.")
                    return False # Stop training
                    
            self.episode_count += 1
            
        return True

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function check
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

def main():
    # Archive first!
    archive_old_runs("dqn")

    # 1. Create Environment
    env = gym.make('SimpleCausalIntersection-v0', render_mode='rgb_array')
    
    # 2. Add Wrappers
    env = OverlayWrapper(env) # Adds text to render()
    env = Monitor(env) # Tracks stats for SB3
    
    # 3. Add Video Recorder
    # Trigger: Record every 30th episode
    video_folder = 'videos_dqn_curved'
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
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(1e-4), # Lower LR
        buffer_size=200000, # Larger buffer
        learning_starts=5000, # More warmup
        batch_size=128, # Larger batch
        tau=0.05, # Soft update
        gamma=0.98,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=5000, # More stable target
        exploration_fraction=0.3, # More exploration (30%)
        exploration_final_eps=0.02,
        verbose=1
    )
    
    # Callback
    viz_callback = VizCallback(viz_freq=100)
    
    # Train
    # User requested threshold ~40-50s driving.
    # At 30 FPS = 1500 steps.
    # If reward ~0.5 per step (speed dependent), total ~750.
    # Let's target 750.0 reward or just train longer.
    
    print("Starting DQN training (Curved Road)...")
    print("Videos will be saved to ./videos_dqn_curved every 30 episodes")
    print("Visualization window will appear every 100 episodes.")
    
    model.learn(total_timesteps=600000, callback=viz_callback) # Increased timesteps
    
    # 6. Save
    model.save("dqn_simple_causal_curved")
    print("Model saved to dqn_simple_causal_curved.zip")
    
    # 7. Plot Rewards
    rewards = viz_callback.all_rewards
    if rewards:
        final_reward = rewards[-1]
        print(f"Final Reward: {final_reward}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title("DQN Episode Rewards (Curved Road)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig("reward_plot_dqn_curved.png")
        print("Reward plot saved to reward_plot_dqn_curved.png")
    
    env.close()
    if plt.get_fignums():
        plt.close('all')

if __name__ == '__main__':
    main()
