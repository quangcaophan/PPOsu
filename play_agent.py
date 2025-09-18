"""
Script để chạy agent đã được train để chơi osu! mania
"""

from stable_baselines3 import PPO
from improved_game_env import OsuManiaEnv
import time
import cv2
import numpy as np

# Configuration
PLAY_AREA = {'top': 230, 'left': 710, 'width': 490, 'height': 750}
NUM_KEYS = 4

def load_and_play_agent(model_path="models/best_osu_model.zip"):
    """Load trained model and play osu! mania"""
    
    print("=== Osu! Mania AI Player ===")
    print(f"Loading model from: {model_path}")
    
    # Create environment with visualization
    env = OsuManiaEnv(
        play_area=PLAY_AREA,
        num_keys=NUM_KEYS,
        show_window=True
    )
    
    try:
        # Load the trained model
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully!")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_with_visualization.py")
        return
    
    print("\nInstructions:")
    print("1. Open osu! mania and select a song")
    print("2. Press any key to start the AI agent")
    print("3. Press 'q' in the visualization window to quit")
    print("4. Press 'r' to reset/restart")
    
    input("Press Enter when ready...")
    
    # Main game loop
    try:
        episode = 1
        while True:
            print(f"\n--- Episode {episode} ---")
            print("Resetting environment... Start your song now!")
            
            obs, info = env.reset()
            total_reward = 0
            step_count = 0
            
            done = False
            while not done:
                # Predict action using trained model
                action, _states = model.predict(obs, deterministic=True)
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # Print stats every 200 steps
                if step_count % 200 == 0:
                    activity = info.get('activity_score', 0)
                    keys = info.get('keys_pressed', [])
                    key_str = ''.join([k.upper() if keys[i] else '-' for i, k in enumerate(['d', 'f', 'j', 'k'])])
                    
                    print(f"Step {step_count:4d} | Reward: {reward:6.2f} | Total: {total_reward:8.2f} | Activity: {activity:.3f} | Keys: {key_str}")
                
                # Handle user input (check for quit/restart)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    done = True
                    env.close()
                    return
                elif key == ord('r'):
                    print("Restarting...")
                    break
            
            print(f"Episode {episode} finished!")
            print(f"Total steps: {step_count}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Average reward per step: {total_reward/max(step_count, 1):.3f}")
            
            # Ask user if they want to continue
            print("\nOptions:")
            print("- Press Enter to play another song")
            print("- Type 'q' and press Enter to quit")
            user_input = input("Your choice: ").strip().lower()
            
            if user_input == 'q':
                break
            
            episode += 1
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    except Exception as e:
        print(f"Error during gameplay: {e}")
    
    finally:
        env.close()
        print("Environment closed. Goodbye!")

def analyze_model_performance(model_path="models/best_osu_model.zip", num_test_episodes=5):
    """Analyze the performance of trained model"""
    
    print("=== Model Performance Analysis ===")
    
    env = OsuManiaEnv(
        play_area=PLAY_AREA,
        num_keys=NUM_KEYS,
        show_window=False  # No visualization for analysis
    )
    
    try:
        model = PPO.load(model_path, env=env)
        
        episode_rewards = []
        episode_lengths = []
        action_distributions = []
        
        for episode in range(num_test_episodes):
            print(f"Testing episode {episode + 1}/{num_test_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            episode_actions = []
            
            done = False
            while not done and step_count < 2000:  # Limit for analysis
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                episode_actions.append(action)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            action_distributions.append(episode_actions)
        
        # Print analysis
        print("\n=== Analysis Results ===")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
        
        # Action distribution analysis
        all_actions = [action for episode in action_distributions for action in episode]
        unique_actions, counts = np.unique(all_actions, return_counts=True)
        
        print("\nAction distribution:")
        for action, count in zip(unique_actions, counts):
            percentage = (count / len(all_actions)) * 100
            # Convert action to key combination
            keys = [bool((action >> i) & 1) for i in range(NUM_KEYS)]
            key_str = ''.join(['D' if keys[0] else '-',
                             'F' if keys[1] else '-',
                             'J' if keys[2] else '-',
                             'K' if keys[3] else '-'])
            print(f"  {key_str}: {percentage:5.1f}% ({count:4d} times)")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    finally:
        env.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "models/best_osu_model.zip"
            analyze_model_performance(model_path)
        else:
            load_and_play_agent(sys.argv[1])
    else:
        load_and_play_agent()