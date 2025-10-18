"""
Training modules for PPO Osu! Mania agent.
"""

# Conditional imports to avoid dependency issues during testing
try:
    from .trainer import PPOTrainer
    from .callbacks import (
        SongFinishedEvalCallback,
        CurriculumCallback,
        BehaviorMonitorCallback,
        LearningRateScheduler
    )
    __all__ = [
        "PPOTrainer",
        "SongFinishedEvalCallback", 
        "CurriculumCallback",
        "BehaviorMonitorCallback",
        "LearningRateScheduler"
    ]
except ImportError:
    # If dependencies aren't installed, just define the module structure
    __all__ = [
        "PPOTrainer",
        "SongFinishedEvalCallback", 
        "CurriculumCallback",
        "BehaviorMonitorCallback",
        "LearningRateScheduler"
    ]
