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
FRAME_SIZE = 64
VISUALIZATION_SIZE = 420
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS  # 0.016667 seconds

# OCR and performance settings
OCR_INTERVAL = 1  # OCR every 1 second for better performance
MAX_STEPS_DEFAULT = 15000
ACTIVITY_THRESHOLD = 0.005
GAME_END_TIMEOUT = 20.0

# Performance monitoring
PERFORMANCE_BUFFER_SIZE = 1000
MONITOR_INTERVAL = 0.5

# Reward system parameters
MISS_PENALTY = -5.0
KEY_SPAM_PENALTY = -0.01
IDLE_PENALTY = -0.05
MENU_KEY_PENALTY = -0.5
MENU_IDLE_REWARD = 0.1

# File paths
DEFAULT_CONFIG_DIR = "config"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_TEMPLATE_DIR = "templates"
