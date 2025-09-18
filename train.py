from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import os

# Import your improved environment
from improved_game_env import OsuManiaEnv

# Configuration
PLAY_AREA = {'top': 230, 'left': 710, 'width': 490, 'height': 750}
NUM_KEYS = 4

class OsuTrainingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.best_reward = -float('inf')
        
    def _on_step(self) -> bool:
        # Log episode statistics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Log to tensorboard
                self.logger.record('episode/reward', episode_reward)
                self.logger.record('episode/length', episode_length)
                self.logger.record('episode/reward_mean_100', np.mean(self.episode_rewards))
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.model.save("models/best_osu_model")
                    print(f"New best reward: {episode_reward:.2f}")
        
        return True

def test_agent(model, env, num_episodes=3):
    """Test the trained agent and visualize performance"""
    print("\n=== Testing Trained Agent ===")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}")
        print("Starting in 5 seconds... Make sure osu! is ready!")
        time.sleep(5)
        
        done = False
        while not done and step_count < 1000:  # Limit test episodes
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                activity_score = info.get('activity_score', 0)
                keys_pressed = info.get('keys_pressed', [])
                print(f"Step {step_count}: Reward={reward:.2f}, Activity={activity_score:.3f}, Keys={keys_pressed}")
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished: Total Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

def setup_training_environment():
    """Setup directories and logging"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    # Setup custom logger
    new_logger = configure("logs/", ["stdout", "csv", "tensorboard"])
    return new_logger

def main():
    print("=== Osu! Mania RL Training Script ===")
    print("Please make sure osu! is open and ready!")
    
    # Check device
    if torch.dml.is_available():
        device = "dml"
        print("Using DirectML (AMD GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Setup training environment
    logger = setup_training_environment()
    
    # Create environment with visualization enabled
    print("\nCreating environment...")
    env = OsuManiaEnv(
        play_area=PLAY_AREA, 
        num_keys=NUM_KEYS, 
        show_window=True  # Enable visualization
    )
    
    # Check if we have a saved model to continue training
    model_path = "models/latest_osu_model.zip"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        model.set_logger(logger)
    else:
        print("Creating new model...")
        # Create new model with optimized hyperparameters for osu!
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device=device,
            # Optimized hyperparameters for rhythm game
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="tensorboard_logs/",
        )
        model.set_logger(logger)
    
    # Setup callback
    callback = OsuTrainingCallback(verbose=1)
    
    print("\nTraining will start in 10 seconds...")
    print("Please:")
    print("1. Make sure osu! mania is open")
    print("2. Select a song (preferably an easier one for initial training)")
    print("3. The AI will start learning to play!")
    print("4. You can watch the visualization window to see what the AI sees")
    
    time.sleep(10)
    
    try:
        # Training loop with periodic saves
        total_timesteps = 500_000
        save_interval = 50_000
        
        for i in range(0, total_timesteps, save_interval):
            remaining_steps = min(save_interval, total_timesteps - i)
            
            print(f"\nTraining steps {i} to {i + remaining_steps}")
            model.learn(
                total_timesteps=remaining_steps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Save model periodically
            model.save(f"models/osu_model_step_{i + remaining_steps}")
            model.save("models/latest_osu_model")
            print(f"Model saved at step {i + remaining_steps}")
            
            # Quick test run
            if (i + remaining_steps) % (save_interval * 2) == 0:  # Test every 2 intervals
                print("\nTesting current model...")
                test_agent(model, env, num_episodes=1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user...")
        model.save("models/interrupted_osu_model")
        print("Model saved!")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        model.save("models/error_backup_osu_model")
        print("Backup model saved!")
    
    finally:
        # Final test
        print("\nFinal testing...")
        final_score = test_agent(model, env, num_episodes=3)
        
        # Save final model
        model.save("models/final_osu_model")
        print(f"\nTraining completed! Final average score: {final_score:.2f}")
        print("Models saved in 'models/' directory")
        
        env.close()

if __name__ == '__main__':
    main()