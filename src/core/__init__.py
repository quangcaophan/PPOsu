"""
Core modules for the PPO Osu! Mania agent.
"""

from .config_manager import ConfigManager
from .logger import setup_logger

__all__ = ["ConfigManager", "setup_logger"]
