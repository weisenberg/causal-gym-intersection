import gymnasium as gym
import gym_causal_intersection
import pygame
import time
import numpy as np

def main():
    print("Creating SimpleCausalIntersection-v0...")
    env = gym.make("SimpleCausalIntersection-v0", render_mode="human")
    
    print("Action Space:", env.action_space)
    print("Observation Space:", env.observation_space)
    
    obs, info = env.reset()
    print("Reset complete.")
    print("Initial Observation:", obs)
    print("Info:", info)
    
    # Run a loop
    print("\nManual Control Enabled:")
    print("  W: Accelerate")
    print("  S: Brake")
    print("  A: Steer Left")
    print("  D: Steer Right")
    print("  No Key: Idle/Coast")
    
    running = True
    while running:
        # Default action: 0 (Idle)
        action = 0 
        
        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Handle Keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
            
        if keys[pygame.K_w]:
            action = 1 # Accel
        elif keys[pygame.K_s]:
            action = 2 # Brake
            
        # Independent steering check (if we supported combined actions, but we are Discrete)
        # Discrete(5): 0=Idle, 1=Accel, 2=Brake, 3=Left, 4=Right
        # We have to prioritize. Steering usually requires moving? 
        # But this env separates them. 
        # Let's prioritize Steering if pressed (car maintains momentum), or we need to switch/alternate?
        # The SimpleEnv is Discrete(5). You can't Accel AND Steer.
        # So we prioritize Steering if pressed, otherwise Accel/Brake.
        
        if keys[pygame.K_a]:
            action = 4 # Right (Swapped to fix reversal)
        elif keys[pygame.K_d]:
            action = 3 # Left (Swapped to fix reversal)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Finished. Reward: {reward:.2f}")
            obs, info = env.reset()
            
        # Cap FPS
        time.sleep(1/30.0)
        
    env.close()

if __name__ == "__main__":
    main()
