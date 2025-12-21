import gymnasium as gym
import gym_causal_intersection
import pygame
import time

def main():
    print("Creating SimpleCausalIntersection-v0 for Manual Control...")
    print("Controls:")
    print("  W: Accelerate")
    print("  S: Brake")
    print("  A: Steer Left")
    print("  D: Steer Right")
    print("  ESC: Exit")
    
    env = gym.make("SimpleCausalIntersection-v0", render_mode="human")
    obs, info = env.reset()
    
    running = True
    while running:
        # Default action: Idle
        action = 0 
        
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Handle Continuous Key Presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = 1 # Accel
        elif keys[pygame.K_s]:
            action = 2 # Brake
        elif keys[pygame.K_a]:
            action = 4 # Right (Swapped based on user feedback)
        elif keys[pygame.K_d]:
            action = 3 # Left (Swapped based on user feedback)
            
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward for display (env handles self.episode_reward internally for render, but good to track)
        # The env.episode_reward is updated in step() if we use the parent's logic, 
        # but SimpleCausalIntersectionEnv overrides step() and might not be updating self.episode_reward.
        # Let's check: SimpleCausalIntersectionEnv.step() does NOT update self.episode_reward.
        # We should update it in the env class or here. 
        # Actually, let's update the env class to track it for rendering.
        # But for now, let's just rely on the env.reset() to clear it if we fix the env.
        
        if terminated or truncated:
            print(f"Episode Finished. Reward: {reward}")
            obs, info = env.reset()
            # Small pause to see the end state?
            # time.sleep(0.5) 
        
    env.close()
    print("Exited.")

if __name__ == "__main__":
    main()
