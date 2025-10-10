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
FRAME_SIZE = 96
VISUALIZATION_SIZE = 420
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS  # 0.016667 seconds

# OCR and performance settings
OCR_INTERVAL = 0.5  # OCR every 1 second for better performance
ACTIVITY_THRESHOLD = 0.005 # This is no longer used for game state, but might be useful for other things

# Performance monitoring
PERFORMANCE_BUFFER_SIZE = 1000
MONITOR_INTERVAL = 0.5

# File paths
DEFAULT_CONFIG_DIR = "config"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_TEMPLATE_DIR = "templates"
