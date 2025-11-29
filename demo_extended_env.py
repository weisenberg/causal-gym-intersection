"""Interactive demo of the UrbanCausalIntersectionExtended-v0 environment."""
import gymnasium as gym
import gym_causal_intersection
import numpy as np
import time
import pygame

def demo_extended_environment():
    # Create the extended environment with rendering
    env = gym.make('UrbanCausalIntersectionExtended-v0', render_mode='human')
    
    print("=" * 70)
    print("UrbanCausalIntersectionExtended-v0 Environment Demo")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Large map: 1600x1600 with 3x3 grid of intersections (9 intersections)")
    print("  - Red car: Your vehicle (spawns on right side of road)")
    print("  - Blue cars: NPC vehicles that follow traffic rules")
    print("  - Blue circles: Pedestrians using zebra crossings")
    print("  - Red circles: Jaywalking pedestrians")
    print("  - White stripes: Zebra crossings at each intersection")
    print("  - Traffic lights: Colored circles at intersections")
    print("  - Camera: Follows your car (centered view)")
    print("\nEpisode Rules:")
    print("  - Episode continues indefinitely until:")
    print("    * Car hits a pedestrian (collision)")
    print("    * Car hits an NPC car (collision)")
    print("    * Car drives off-screen (out of bounds)")
    print("  - NPC cars stop at red lights and avoid pedestrians")
    print("  - Pedestrians wait at red lights when using crossings")
    print("\nControls (WASD) - Click the window to focus, then use keys:")
    print("  W: Accelerate forward")
    print("  S: Brake (reverse)")
    print("  A: Turn Left")
    print("  D: Turn Right")
    print("  (No key pressed): Idle")
    print("\nTip: You can combine keys (e.g., W+A to accelerate while turning left)")
    print("     Camera follows your car. Explore the city!")
    print("Press ESC, Ctrl+C, or close window to stop")
    print("=" * 70)
    
    # Reset with default settings (low traffic/peds for demo)
    obs, info = env.reset(options={"traffic_density": "low", "pedestrian_density": "low"})
    
    print(f"\nEnvironment reset!")
    print(f"  Car position: ({obs[0]:.0f}, {obs[1]:.0f})")
    print(f"  Car heading: {obs[4]:.2f} radians")
    print(f"  Distance to nearest intersection: {obs[5]:.1f}")
    print(f"  Initial pedestrians: {info.get('num_pedestrians', 0)}")
    print(f"  Initial NPC cars: {info.get('num_npc_cars', 0)}")
    print(f"  Number of intersections: {len(env.unwrapped.intersections)}")
    print("\n" + "="*70)
    print("IMPORTANT: Click on the pygame window to focus it for keyboard input!")
    print("="*70)
    print("\nStarting simulation...\n")
    
    step_count = 0
    
    try:
        while True:  # Run indefinitely until termination
            # Handle keyboard input for manual control
            action = 0  # Default: Idle
            
            # Check for pygame events (handle window close and ESC key)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            
            # Check for currently held keys (for continuous control)
            keys = pygame.key.get_pressed()
            
            # Continuous action: [acceleration, steering]
            # acceleration: -1.0 (brake) to 1.0 (accelerate)
            # steering: -1.0 (left) to 1.0 (right)
            accel = 0.0
            steer = 0.0
            
            if keys[pygame.K_w]:
                accel = 1.0  # Accelerate forward
            elif keys[pygame.K_s]:
                accel = -1.0  # Brake (until stop, no reverse)
            
            if keys[pygame.K_a]:
                steer = 1.0  # Turn left (swapped sign)
            elif keys[pygame.K_d]:
                steer = -1.0  # Turn right (swapped sign)
            
            action = np.array([accel, steer], dtype=np.float32)
            
            # Take a step
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print reward every step
            num_peds = info.get('num_pedestrians', 0)
            num_cars = info.get('num_npc_cars', 0)
            print(f"Step {step_count}: Reward: {reward:.3f} | "
                  f"Agent: ({obs[0]:.0f}, {obs[1]:.0f}) | "
                  f"Peds: {num_peds}, NPCs: {num_cars}")
            
            # Check for termination
            if terminated:
                reason = "Collision with pedestrian!" if info.get('collision', False) else "Collision with NPC car or off-screen!"
                print(f"\nEpisode terminated at step {step_count}")
                print(f"Reason: {reason}")
                print(f"Final reward: {reward:.2f}")
                
                # Reset for new episode (random properties, but low density)
                print("\nStarting new episode...")
                obs, info = env.reset(options={"traffic_density": "low", "pedestrian_density": "low"})
                print(f"  Car position: ({obs[0]:.0f}, {obs[1]:.0f})")
                print(f"  Car heading: {obs[4]:.2f} radians")
                print(f"  Initial pedestrians: {info.get('num_pedestrians', 0)}")
                print(f"  Initial NPC cars: {info.get('num_npc_cars', 0)}")
                step_count = 0
            
            # Small delay to make it visible
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    
    env.close()
    print("\nDemo completed!")

if __name__ == "__main__":
    demo_extended_environment()

