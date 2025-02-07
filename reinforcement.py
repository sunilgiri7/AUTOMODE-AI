from stable_baselines3 import PPO
import gymnasium as gym  # Updated import
import numpy as np
from typing import List, Dict, Any, Optional

class ReinforcementLearner:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        self.initialized = False

    def train(self, feedback_log: List[Dict[str, Any]], reward_fn: callable) -> Optional[PPO]:
        if not feedback_log:
            return None

        try:
            # Filter valid feedback entries
            valid_feedback = [
                entry for entry in feedback_log 
                if entry.get("user_feedback") is not None
            ]

            if not valid_feedback:
                return None

            # Convert feedback to normalized rewards
            rewards = np.array([
                reward_fn(entry) for entry in valid_feedback
            ])
            
            if len(rewards) > 0:
                # Normalize rewards
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                
                # Train the model
                if not self.initialized:
                    self.model = PPO("MlpPolicy", self.env, verbose=0)
                    self.initialized = True

                self.model.learn(total_timesteps=100 * len(rewards))
                return self.model

        except Exception as e:
            print(f"Error in reinforcement training: {e}")
            return None

        return None

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()