"""
PPO trainer for osu!mania agent.
Clean, modular training system with comprehensive error handling.
"""

import os
import time
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces as gym_spaces

from ..core.config_manager import ConfigManager, AgentConfig
from ..core.logger import setup_colored_logger, get_logger
from ..environment.mania_env import OsuManiaEnv
from .callbacks import (
    SongFinishedEvalCallback,
    CurriculumCallback,
    BehaviorMonitorCallback,
    LearningRateScheduler,
    PerformanceMonitorCallback
)


class PPOTrainer:
    """
    Complete PPO training system for osu!mania agent.
    
    Features:
    - Clean configuration management
    - Comprehensive logging
    - Graceful error handling
    - Modular callback system
    - Automatic checkpointing
    """
    
    def __init__(
        self,
        config_name: str,
        config_dir: str = "config",
        show_eval_window: bool = False
    ):
        """
        Initialize PPO trainer.
        
        Args:
            config_name: Name of configuration file (without .json)
            config_dir: Directory containing config files
            show_eval_window: Whether to show evaluation window
        """
        self.config_name = config_name
        self.config_dir = config_dir
        self.show_eval_window = show_eval_window
        
        # Initialize components
        self.config_manager = ConfigManager(config_dir)
        self.logger = setup_colored_logger("trainer")
        
        # Training state
        self.config: Optional[AgentConfig] = None
        self.model: Optional[PPO] = None
        self.env: Optional[OsuManiaEnv] = None
        self.eval_env: Optional[OsuManiaEnv] = None
        self.callbacks: List = []
        
        # Paths
        self.model_dir: Optional[Path] = None
        self.log_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.tensorboard_dir: Optional[Path] = None
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def load_config(self) -> AgentConfig:
        """Load and validate configuration."""
        try:
            self.config = self.config_manager.load_config(self.config_name)
            
            # Validate configuration
            if not self.config_manager.validate_config(self.config):
                raise ValueError("Configuration validation failed")
            
            self.logger.info(f"Loaded configuration: {self.config_name}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def setup_paths(self) -> None:
        """Setup directory paths for training."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Create run ID
        run_id = f"{self.config.mode}_{self.config.num_keys}k"
        
        # Setup directories
        self.model_dir = Path("models") / self.config.mode / f"{self.config.num_keys}k"
        self.log_dir = Path("logs") / run_id
        self.checkpoint_dir = Path("checkpoints") / run_id
        self.tensorboard_dir = Path("tensorboard_logs")
        
        # Create directories
        for directory in [self.model_dir, self.log_dir, self.checkpoint_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Setup paths for run: {run_id}")
    
    def create_environments(self) -> None:
        """Create training and evaluation environments."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        self.logger.info("Creating environments...")
        
        # Training environment
        self.env = OsuManiaEnv(
            config=self.config,
            show_window=True,
            run_id=f"{self.config.mode}_{self.config.num_keys}k",
            is_eval_env=False
        )
        self.env = Monitor(self.env)
        
        # Evaluation environment
        self.eval_env = OsuManiaEnv(
            config=self.config,
            show_window=self.show_eval_window,
            run_id=f"{self.config.mode}_{self.config.num_keys}k_eval",
            is_eval_env=True
        )
        self.eval_env = Monitor(self.eval_env)
        
        self.logger.info("Environments created successfully")
    
    def create_model(self) -> None:
        """Create or load PPO model."""
        if not self.config or not self.env:
            raise RuntimeError("Configuration and environment must be loaded first")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device.upper()}")
        
        # Model path
        latest_model_path = self.model_dir / "latest_model.zip"
        
        # Load existing model or create new one
        if latest_model_path.exists():
            self.logger.info(f"Loading existing model: {latest_model_path}")
            try:
                self.model = PPO.load(
                    str(latest_model_path), 
                    env=self.env, 
                    device=device
                )
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.logger.info("Creating new model instead...")
                self._create_new_model(device)
        else:
            self.logger.info("Creating new PPO model...")
            self._create_new_model(device)
    
    def _create_new_model(self, device: str) -> None:
        """Create a new PPO model."""
        if not self.config or not self.env:
            raise RuntimeError("Configuration and environment must be loaded first")
        
        # Extract PPO parameters
        ppo_params = self.config.training_params.ppo_params

        # Determine policy with safety fallback for observation type
        requested_policy = self.config.training_params.policy
        policy_to_use = requested_policy
        if requested_policy == "MultiInputPolicy" and not isinstance(self.env.observation_space, gym_spaces.Dict):
            self.logger.warning(
                "Requested MultiInputPolicy but observation_space is not Dict; falling back to CnnPolicy"
            )
            policy_to_use = "CnnPolicy"

        # Auto-configure policy kwargs based on observation scaling
        policy_kwargs: Dict[str, Any] = {}
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym_spaces.Box) and len(obs_space.shape) == 3:
            try:
                max_high = float(np.max(obs_space.high))
                min_low = float(np.min(obs_space.low))
                is_float = np.issubdtype(obs_space.dtype, np.floating)
                # If frames are already normalized to [0, 1] floats, disable image normalization
                if is_float and min_low >= 0.0 and max_high <= 1.0:
                    policy_kwargs["normalize_images"] = False
                    self.logger.info("Detected normalized float image observations; setting normalize_images=False")
            except Exception:
                # Be conservative if any check fails
                pass

        self.model = PPO(
            policy=policy_to_use,
            env=self.env,
            verbose=1,
            device=device,
            tensorboard_log=str(self.tensorboard_dir),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            **ppo_params.__dict__
        )
        
        self.logger.info("New PPO model created")
    
    def setup_callbacks(self) -> None:
        """Setup training callbacks."""
        if not self.config or not self.model or not self.eval_env:
            raise RuntimeError("Configuration, model, and eval environment must be loaded first")
        
        self.callbacks = []
        
        # 1. Checkpoint callback
        self.callbacks.append(
            CheckpointCallback(
                save_freq=self.config.training_params.callback_params.checkpoint_save_freq,
                name_prefix=f"{self.config.mode}_{self.config.num_keys}k",
                save_path=str(self.checkpoint_dir)
            )
        )
        
        # 2. Song-finished evaluation callback
        self.callbacks.append(
            SongFinishedEvalCallback(
                eval_env=self.eval_env,
                best_model_save_path=str(self.model_dir / "best_model"),
                n_eval_episodes=self.config.training_params.callback_params.n_eval_episodes,
                min_songs_between_eval=self.config.training_params.callback_params.min_songs_between_eval,
                verbose=1
            )
        )
        
        # 3. Curriculum learning callback
        self.callbacks.append(
            CurriculumCallback(env=self.env, verbose=1)
        )
        
        # 4. Behavior monitoring callback
        self.callbacks.append(
            BehaviorMonitorCallback(check_freq=2000, verbose=1)
        )
        
        # 5. Learning rate scheduler
        self.callbacks.append(
            LearningRateScheduler(verbose=1)
        )
        
        # 6. Performance monitoring callback
        self.callbacks.append(
            PerformanceMonitorCallback(check_freq=1000, verbose=1)
        )
        
        self.logger.info(f"Setup {len(self.callbacks)} callbacks")
    
    def train(self, total_timesteps: Optional[int] = None) -> None:
        """Run training."""
        if not all([self.config, self.model, self.env, self.callbacks]):
            raise RuntimeError("Training not properly initialized")
        
        # Use provided timesteps or config default
        if total_timesteps is None:
            total_timesteps = self.config.training_params.total_timesteps
        
        self.logger.info(f"Starting training for {total_timesteps:,} timesteps")
        self.logger.info(f"Model directory: {self.model_dir}")
        self.logger.info(f"TensorBoard logs: {self.tensorboard_dir}")
        self.logger.info("Press Ctrl+C for safe exit")
        
        # Wait a moment for user to read
        time.sleep(3)
        
        start_time = time.time()
        
        try:
            # Run training
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                reset_num_timesteps=(not (self.model_dir / "latest_model.zip").exists()),
                progress_bar=True,
                tb_log_name=f"{self.config.mode}_{self.config.num_keys}k"
            )
            
            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            self.model.save(str(final_model_path))
            self.logger.info(f"Training completed! Final model saved: {final_model_path}")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
            
            # Log training time
            hours = (time.time() - start_time) / 3600
            self.logger.info(f"Total training time: {hours:.1f} hours")
    
    def cleanup(self) -> None:
        """Cleanup training resources."""
        self.logger.info("Cleaning up training resources...")
        
        # Save latest model
        if self.model:
            try:
                latest_model_path = self.model_dir / "latest_model.zip"
                self.model.save(str(latest_model_path))
                self.logger.info(f"Saved latest model: {latest_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save latest model: {e}")
        
        # Close environments
        for env_name, env in [("env", self.env), ("eval_env", self.eval_env)]:
            if env:
                try:
                    env.close()
                    self.logger.info(f"Closed {env_name}")
                except Exception as e:
                    self.logger.error(f"Failed to close {env_name}: {e}")
        
        self.logger.info("Cleanup completed")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_training(
        self, 
        total_timesteps: Optional[int] = None
    ) -> None:
        """
        Complete training pipeline.
        
        Args:
            total_timesteps: Number of timesteps to train (uses config default if None)
        """
        try:
            # Load configuration
            self.load_config()
            
            # Setup paths
            self.setup_paths()
            
            # Create environments
            self.create_environments()
            
            # Create model
            self.create_model()
            
            # Setup callbacks
            self.setup_callbacks()
            
            # Run training
            self.train(total_timesteps)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.cleanup()
            raise
