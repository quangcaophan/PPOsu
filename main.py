"""
Main entry point for PPO Osu! Mania agent.
Refactored version with clean architecture.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.trainer import PPOTrainer
from src.core.logger import setup_colored_logger
from src.core.config_manager import ConfigManager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PPO Osu! Mania AI Agent - Refactored Version"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Configuration name (without .json extension)"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=None, 
        help="Override total timesteps from config"
    )
    parser.add_argument(
        "--show-eval", 
        action="store_true", 
        help="Show evaluation environment window"
    )
    parser.add_argument(
        "--config-dir", 
        type=str, 
        default="config", 
        help="Configuration directory"
    )
    parser.add_argument(
        "--auto-select-songs",
        action="store_true",
        help="Automatically select random songs during training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_colored_logger("main")
    
    try:
        # Check if config exists
        config_manager = ConfigManager(args.config_dir)
        if args.config not in config_manager.list_configs():
            logger.error(f"Configuration '{args.config}' not found in {args.config_dir}")
            logger.info(f"Available configs: {config_manager.list_configs()}")
            sys.exit(1)
        
        # Create trainer
        trainer = PPOTrainer(
            config_name=args.config,
            config_dir=args.config_dir,
            show_eval_window=args.show_eval,
            auto_select_songs=args.auto_select_songs
        )
        
        # Run training
        trainer.run_training(total_timesteps=args.timesteps)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()