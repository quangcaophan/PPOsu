"""
Constants for osu!mania Environment
All shared constants and configurations
"""

# Key mappings for different osu!mania modes
KEY_MAPPINGS = {
    4: ['d', 'f', 'j', 'k'],
    5: ['d', 'f', 'space', 'j', 'k'],
    6: ['s', 'd', 'f', 'j', 'k', 'l'],
    7: ['s', 'd', 'f', 'space', 'j', 'k', 'l']
}

# Frame and visualization settings
FRAME_SIZE = 84
VISUALIZATION_SIZE = 420
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS  # 0.016667 seconds

# Performance monitoring
VIS_SIZE = 280

# ================== DIRECTORY AND FILE PATHS ==================
# Base Directories
MODELS_DIR = "models"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_DIR = "tensorboard_logs"

# File/Folder Names
LATEST_MODEL_NAME = "latest_model.zip"
FINAL_MODEL_NAME = "final_model.zip"
BEST_MODEL_DIR_NAME = "best_model"


# ================== CONFIGURATION KEYS ==================
# Top-level keys
CONFIG_MODE = 'mode'
CONFIG_PLAY_AREA = 'play_area'
CONFIG_NUM_KEYS = 'num_keys'
CONFIG_TRAINING_PARAMS = 'training_params'

# 'training_params' sub-keys
CONFIG_PPO_PARAMS = 'ppo_params'
CONFIG_CALLBACK_PARAMS = 'callback_params'
CONFIG_TOTAL_TIMESTEPS = 'total_timesteps'
CONFIG_POLICY = 'policy'

# 'callback_params' sub-keys
CALLBACK_CHECKPOINT_FREQ = "checkpoint_save_freq"
CALLBACK_EVAL_FREQ = "eval_freq"
CALLBACK_N_EVAL_EPISODES = "n_eval_episodes"


# ================== REWARD PARAMETERS ==================
# These can be overridden in the config.json
DEFAULT_REWARD_PARAMS = {
    "living_penalty": -0.01,
    "action_cost_penalty": -0.005,
    "hit_geki_reward": 5.0,
    "hit_300_reward": 3.0,
    "hit_100_reward": 1.0,
    "hit_50_penalty": -0.5,
    "miss_penalty": -1.0,
    "combo_break_penalty": -0.5,
    "combo_increase_reward": 0.1,
    "combo_milestone_50": 10.0,
    "combo_milestone_100": 20.0,
    "combo_milestone_200": 40.0,
    "accuracy_change_multiplier": 100.0,
    "idle_penalty": -0.02,
}


# ================== DEFAULT TRAINING VALUES ==================
DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_CHECKPOINT_FREQ = 10000
DEFAULT_EVAL_FREQ = 5000
DEFAULT_N_EVAL_EPISODES = 5
DEFAULT_POLICY = "CnnPolicy"
