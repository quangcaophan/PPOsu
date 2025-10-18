"""
Utility modules for PPO Osu! Mania agent.
"""

# Conditional imports to avoid dependency issues during testing
try:
    from .visualization import VisualizationManager
    from .performance import PerformanceMonitor
    __all__ = ["VisualizationManager", "PerformanceMonitor"]
except ImportError:
    # If dependencies aren't installed, just define the module structure
    __all__ = ["VisualizationManager", "PerformanceMonitor"]
