import gymnasium as gym
import numpy as np

class SafetyWrapper(gym.Wrapper):
    """
    A wrapper that implements a safety shield.
    For now, it acts as a pass-through, but can be extended to override actions
    based on Time-To-Collision (TTC) calculations.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        return self.env.step(action)
