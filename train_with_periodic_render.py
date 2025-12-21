import gymnasium as gym
import gym_causal_intersection
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os

class StopTrainingOnRewardThreshold(BaseCallback):
    """
    Callback that stops training when the average reward exceeds a threshold.
    """
    def __init__(self, reward_threshold=1200, verbose=0):
        super(StopTrainingOnRewardThreshold, self).__init__(verbose)
        self.reward_threshold = reward_threshold
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check for episode completions
        dones = self.locals['dones']
        num_dones = np.sum(dones)
        
        if num_dones > 0:
            self.episode_count += num_dones
            
            # Check stopping condition
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0 and len(ep_info_buffer) >= 100:
                mean_reward = np.mean([ep_info['r'] for ep_info in ep_info_buffer])
                print(f"\n[Episode {self.episode_count}] Mean reward (last 100 eps): {mean_reward:.2f}")
                if mean_reward >= self.reward_threshold:
                    print(f"\n[SUCCESS] Mean reward ({mean_reward:.2f}) exceeded threshold ({self.reward_threshold}). Stopping training!")
                    return False  # Stop training

        return True

def train_and_demonstrate():
    # Configuration
    REWARD_THRESHOLD = 1200  # Stop if mean reward > 1200
    TOTAL_TIMESTEPS = 20000000  # Upper limit
    
    print(f"Starting training (headless)...")
    print(f"- Stop if Mean Reward (last 100 eps) > {REWARD_THRESHOLD}")
    print(f"- Training without rendering for maximum speed")
    
    # 1. Create Training Environment (Headless - Fast)
    train_env = gym.make('UrbanCausalIntersection-v0', render_mode=None)
    
    # 2. Initialize Agent
    model = PPO("MlpPolicy", train_env, verbose=1)
    
    # 3. Initialize Callback
    callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD)
    
    # 4. Train
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60 + "\n")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    train_env.close()
    
    # Save the trained model
    model_path = "models/ppo_intersection_trained.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # 5. Demonstrate the trained agent
    print("\n" + "="*60)
    print("DEMONSTRATION PHASE - Showing trained agent")
    print("="*60 + "\n")
    
    demo_env = gym.make('UrbanCausalIntersection-v0', render_mode="human")
    
    # Run one episode with the trained model
    obs, _ = demo_env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    print("Running demonstration episode...")
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = demo_env.step(action)
        total_reward += reward
        steps += 1
    
    print(f"\nDemonstration complete!")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Steps: {steps}")
    print(f"  Success: {info.get('success', False)}")
    
    demo_env.close()
    print("\nTraining complete.")

if __name__ == "__main__":
    train_and_demonstrate()
