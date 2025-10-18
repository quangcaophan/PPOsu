"""
Refactored agent player for osu!mania.
Clean, modular design with better error handling.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.environment.mania_env import OsuManiaEnv
from src.core.config_manager import ConfigManager
from src.core.logger import setup_colored_logger
from stable_baselines3 import PPO


def play_agent(
    model_path: str = "models/mania/4k/best_model/best_model.zip",
    config_name: str = "mania_4k"
):
    """
    Play with trained agent.
    
    Args:
        model_path: Path to trained model
        config_name: Configuration name
    """
    logger = setup_colored_logger("play_agent")
    
    logger.info("=== Osu! Mania AI Player (Refactored) ===")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_name)
        
        # Create environment
        env = OsuManiaEnv(
            config=config,
            show_window=True,
            run_id="play_agent"
        )
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)
        logger.info("✅ Model loaded successfully!")
        
        # Instructions
        print("\nInstructions:")
        print("1. Open osu! and select a song.")
        print("2. When ready, switch back to this terminal and press Enter.")
        print("3. To stop the agent, focus the visualization window and press 'q'.")
        input("Press Enter to start the agent...")
        
        # Play episodes
        episodes = 0
        while True:
            episodes += 1
            logger.info(f"--- Starting Episode {episodes} ---")
            
            obs, info = env.reset()
            done, truncated = False, False
            total_reward = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            logger.info(f"--- Episode {episodes} Finished ---")
            logger.info(f"Total reward: {total_reward:.2f}")
            
            # Check if user wants to quit
            if done:
                logger.info("Quit signal received from environment. Exiting.")
                break
            
            # Ask if user wants to continue
            continue_playing = input("Play another song? (Y/n): ").strip().lower()
            if continue_playing == 'n':
                break
        
        # Cleanup
        env.close()
        logger.info("Goodbye!")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.info("Please train the model first.")
    except Exception as e:
        logger.error(f"Error during play: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play with trained PPO Osu! Mania agent")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/mania/4k/best_model/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="mania_4k",
        help="Configuration name"
    )
    
    args = parser.parse_args()
    
    play_agent(
        model_path=args.model,
        config_name=args.config
    )


if __name__ == "__main__":
    main()
