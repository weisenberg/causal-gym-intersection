import gymnasium as gym
import gym_causal_intersection
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import os
import time
import multiprocessing

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
                if self.verbose > 0 and self.episode_count % 100 == 0:
                    print(f"\n[Episode {self.episode_count}] Mean reward (last 100 eps): {mean_reward:.2f}")
                
                if mean_reward >= self.reward_threshold:
                    print(f"\n[SUCCESS] Mean reward ({mean_reward:.2f}) exceeded threshold ({self.reward_threshold}). Stopping training!")
                    return False  # Stop training

        return True

def make_env(rank, seed=0, max_npcs=2, max_pedestrians=2):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param max_npcs: (int) limit number of NPCs for faster training
    :param max_pedestrians: (int) limit number of pedestrians
    """
    def _init():
        # Create environment with optimized settings for training
        env = gym.make(
            'UrbanCausalIntersection-v0', 
            render_mode=None,
            max_npcs=max_npcs,
            max_pedestrians=max_pedestrians
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train_optimized():
    # Configuration
    REWARD_THRESHOLD = 1200
    TOTAL_TIMESTEPS = 5_000_000  # 5M steps
    NUM_CPU = min(8, multiprocessing.cpu_count())  # Use up to 8 cores
    
    # Training Environment Settings (Fast Mode)
    MAX_NPCS_TRAIN = 2       # Reduced from default 6
    MAX_PEDS_TRAIN = 2       # Reduced from default 8
    
    print(f"Starting OPTIMIZED training...")
    print(f"- Parallel Environments: {NUM_CPU}")
    print(f"- Stop Threshold: {REWARD_THRESHOLD}")
    print(f"- Training Env: max_npcs={MAX_NPCS_TRAIN}, max_peds={MAX_PEDS_TRAIN}")
    
    # 1. Create Vectorized Environment
    # SubprocVecEnv runs each env in a separate process
    train_env = make_vec_env(
        env_id=make_env(0, max_npcs=MAX_NPCS_TRAIN, max_pedestrians=MAX_PEDS_TRAIN),
        n_envs=NUM_CPU,
        seed=0,
        vec_env_cls=SubprocVecEnv
    )
    
    # Wrap in VecMonitor to log episode rewards/lengths
    train_env = VecMonitor(train_env)
    
    # 2. Initialize Agent with Optimized Hyperparameters
    # These settings are better suited for PPO in continuous control tasks
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,           # More steps per update (better for complex dynamics)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # Encourage exploration
    )
    
    # 3. Initialize Callback
    callback = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)
    
    # 4. Train
    print("\n" + "="*60)
    print("TRAINING PHASE (Optimized)")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds")
    
    train_env.close()
    
    # Save the trained model
    model_path = "models/ppo_intersection_optimized.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # 5. Demonstrate the trained agent (Full Environment)
    print("\n" + "="*60)
    print("DEMONSTRATION PHASE - Showing trained agent (Full Env)")
    print("="*60 + "\n")
    
    # Use full environment for demonstration (default settings)
    demo_env = gym.make('UrbanCausalIntersection-v0', render_mode="human")
    
    # Run a few episodes
    for i in range(3):
        obs, _ = demo_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"Running demonstration episode {i+1}...")
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = demo_env.step(action)
            total_reward += reward
            steps += 1
            
            # Optional: slow down slightly for viewing if needed, 
            # but usually the env render fps handles it.
        
        print(f"  Episode {i+1} Complete: Reward={total_reward:.2f}, Steps={steps}, Success={info.get('success', False)}")
    
    demo_env.close()
    print("\nAll done.")

if __name__ == "__main__":
    train_optimized()
