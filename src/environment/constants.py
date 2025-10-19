"""
Constants and configuration for the osu!mania environment.
"""

from typing import Dict, List

# Key mappings for different osu!mania modes
KEY_MAPPINGS: Dict[int, List[str]] = {
    4: ['d', 'f', 'j', 'k'],
    5: ['d', 'f', 'space', 'j', 'k'],
    6: ['s', 'd', 'f', 'j', 'k', 'l'],
    7: ['s', 'd', 'f', 'space', 'j', 'k', 'l']
}

# Frame and visualization settings
FRAME_SIZE = 128
VISUALIZATION_SIZE = 420
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS  # 0.016667 seconds

# Performance monitoring
VIS_SIZE = 280

# Directory and file paths
MODELS_DIR = "models"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_DIR = "tensorboard_logs"

# File/folder names
LATEST_MODEL_NAME = "latest_model.zip"
FINAL_MODEL_NAME = "final_model.zip"
BEST_MODEL_DIR_NAME = "best_model"

# Configuration keys
CONFIG_MODE = 'mode'
CONFIG_PLAY_AREA = 'play_area'
CONFIG_NUM_KEYS = 'num_keys'
CONFIG_TRAINING_PARAMS = 'training_params'
CONFIG_PPO_PARAMS = 'ppo_params'
CONFIG_CALLBACK_PARAMS = 'callback_params'
CONFIG_TOTAL_TIMESTEPS = 'total_timesteps'
CONFIG_POLICY = 'policy'

# Callback parameter keys
CALLBACK_CHECKPOINT_FREQ = "checkpoint_save_freq"
CALLBACK_EVAL_FREQ = "eval_freq"
CALLBACK_N_EVAL_EPISODES = "n_eval_episodes"

# Default training values
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_FREQ = 10000
DEFAULT_EVAL_FREQ = 5000
DEFAULT_N_EVAL_EPISODES = 5
DEFAULT_POLICY = "CnnPolicy"

# Game state constants (aligned with tosu)
# 5: menu, 2: gameplay, 7: results
GAME_STATE_MENU = 5
GAME_STATE_PLAYING = 2
GAME_STATE_RESULTS = 7

# Performance thresholds
NO_DATA_THRESHOLD = 30  # consecutive frames without data before truncation (~0.5s)
NOT_PLAYING_TERMINATION_STEPS = 30  # consecutive not-playing frames before terminate (~0.5s)
MAX_FRAME_QUEUE_SIZE = 2
