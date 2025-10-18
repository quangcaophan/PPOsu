"""
Environment modules for PPO Osu! Mania agent.
"""

# Conditional imports to avoid dependency issues during testing
try:
    from .mania_env import OsuManiaEnv
    from .reward_calculator import RewardCalculator
    from .frame_processor import FrameProcessor
    __all__ = ["OsuManiaEnv", "RewardCalculator", "FrameProcessor"]
except ImportError:
    # If dependencies aren't installed, just define the module structure
    __all__ = ["OsuManiaEnv", "RewardCalculator", "FrameProcessor"]
