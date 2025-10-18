"""
Configuration management system for PPO Osu! Mania agent.
Handles loading, validation, and management of configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayArea:
    """Play area configuration for screen capture."""
    top: int
    left: int
    width: int
    height: int


@dataclass
class RewardParams:
    """Reward function parameters."""
    living_penalty: float = -0.001
    action_cost_penalty: float = -0.0005
    idle_penalty: float = -0.01
    
    hit_geki_reward: float = 3.0
    hit_300_reward: float = 2.0
    hit_100_reward: float = 1.0
    hit_50_penalty: float = -0.1
    miss_penalty: float = -0.2
    
    combo_break_penalty: float = -0.1
    combo_increase_reward: float = 0.05
    
    combo_milestone_50: float = 5.0
    combo_milestone_100: float = 10.0
    combo_milestone_200: float = 20.0
    
    accuracy_change_multiplier: float = 10.0


@dataclass
class PPOParams:
    """PPO algorithm hyperparameters."""
    n_steps: int = 512
    batch_size: int = 128
    ent_coef: float = 0.15
    learning_rate: float = 0.0001
    clip_range: float = 0.2
    n_epochs: int = 4
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5


@dataclass
class CallbackParams:
    """Training callback parameters."""
    checkpoint_save_freq: int = 5000
    eval_freq: int = 5000
    n_eval_episodes: int = 2
    min_songs_between_eval: int = 3


@dataclass
class TrainingParams:
    """Training configuration parameters."""
    policy: str = "CnnPolicy"
    total_timesteps: int = 100000
    ppo_params: PPOParams = None
    callback_params: CallbackParams = None
    
    def __post_init__(self):
        if self.ppo_params is None:
            self.ppo_params = PPOParams()
        if self.callback_params is None:
            self.callback_params = CallbackParams()


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    mode: str = "mania"
    num_keys: int = 4
    play_area: PlayArea = None
    max_steps: int = 15000
    reward_params: RewardParams = None
    training_params: TrainingParams = None
    timestamp: str = ""
    
    def __post_init__(self):
        if self.play_area is None:
            self.play_area = PlayArea(top=172, left=728, width=470, height=716)
        if self.reward_params is None:
            self.reward_params = RewardParams()
        if self.training_params is None:
            self.training_params = TrainingParams()


class ConfigManager:
    """Manages configuration loading, validation, and saving."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_name: str) -> AgentConfig:
        """Load configuration from file."""
        config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert dict to dataclass
            config = self._dict_to_config(data)
            logger.info(f"Loaded config: {config_name}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def save_config(self, config: AgentConfig, config_name: str) -> None:
        """Save configuration to file."""
        config_path = self.config_dir / f"{config_name}.json"
        
        try:
            # Convert dataclass to dict
            data = self._config_to_dict(config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved config: {config_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save config: {e}")
    
    def create_default_config(self, mode: str, num_keys: int) -> AgentConfig:
        """Create a default configuration for given mode and key count."""
        config = AgentConfig(
            mode=mode,
            num_keys=num_keys,
            timestamp=self._get_timestamp()
        )
        
        # Adjust play area based on key count
        if num_keys == 7:
            config.play_area = PlayArea(top=156, left=731, width=452, height=713)
        
        return config
    
    def validate_config(self, config: AgentConfig) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate required fields
            if not config.mode:
                raise ValueError("Mode is required")
            
            if config.num_keys not in [4, 5, 6, 7]:
                raise ValueError(f"Unsupported key count: {config.num_keys}")
            
            if config.max_steps <= 0:
                raise ValueError("max_steps must be positive")
            
            # Validate play area
            if config.play_area.width <= 0 or config.play_area.height <= 0:
                raise ValueError("Play area dimensions must be positive")
            
            # Validate reward parameters
            if config.reward_params.miss_penalty > 0:
                raise ValueError("Miss penalty should be negative")
            
            logger.info("Config validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AgentConfig:
        """Convert dictionary to AgentConfig dataclass."""
        # Handle nested dataclasses
        if 'play_area' in data:
            data['play_area'] = PlayArea(**data['play_area'])
        
        if 'reward_params' in data:
            data['reward_params'] = RewardParams(**data['reward_params'])
        
        if 'training_params' in data:
            training_data = data['training_params']
            if 'ppo_params' in training_data:
                training_data['ppo_params'] = PPOParams(**training_data['ppo_params'])
            if 'callback_params' in training_data:
                training_data['callback_params'] = CallbackParams(**training_data['callback_params'])
            data['training_params'] = TrainingParams(**training_data)
        
        return AgentConfig(**data)
    
    def _config_to_dict(self, config: AgentConfig) -> Dict[str, Any]:
        """Convert AgentConfig dataclass to dictionary."""
        return asdict(config)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def list_configs(self) -> list:
        """List available configuration files."""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
