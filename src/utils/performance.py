"""
Performance monitoring utilities for the PPO Osu! Mania agent.
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from pathlib import Path

from ..core.logger import get_logger


class PerformanceMonitor:
    """Monitors system and training performance."""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Size of performance history to keep
        """
        self.history_size = history_size
        self.logger = get_logger("performance_monitor")
        
        # Performance metrics
        self.fps_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.reward_history = deque(maxlen=history_size)
        
        # Timing
        self.last_update = time.time()
        self.frame_times = deque(maxlen=60)  # Last 60 frames
        
        # System info
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def update_frame_time(self, frame_time: float) -> None:
        """Update frame processing time."""
        self.frame_times.append(frame_time)
        
        # Calculate FPS
        if len(self.frame_times) > 1:
            avg_frame_time = np.mean(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.fps_history.append(fps)
    
    def update_reward(self, reward: float) -> None:
        """Update reward history."""
        self.reward_history.append(reward)
    
    def update_system_metrics(self) -> Dict[str, float]:
        """Update and return system metrics."""
        current_time = time.time()
        
        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        self.memory_history.append(memory_mb)
        
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        self.cpu_history.append(cpu_percent)
        
        self.last_update = current_time
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "uptime": current_time - self.start_time
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}
        
        # FPS stats
        if self.fps_history:
            stats.update({
                "avg_fps": np.mean(self.fps_history),
                "min_fps": np.min(self.fps_history),
                "max_fps": np.max(self.fps_history),
                "current_fps": self.fps_history[-1] if self.fps_history else 0
            })
        
        # Memory stats
        if self.memory_history:
            stats.update({
                "avg_memory_mb": np.mean(self.memory_history),
                "max_memory_mb": np.max(self.memory_history),
                "current_memory_mb": self.memory_history[-1]
            })
        
        # CPU stats
        if self.cpu_history:
            stats.update({
                "avg_cpu_percent": np.mean(self.cpu_history),
                "max_cpu_percent": np.max(self.cpu_history),
                "current_cpu_percent": self.cpu_history[-1]
            })
        
        # Reward stats
        if self.reward_history:
            recent_rewards = list(self.reward_history)[-100:]  # Last 100 rewards
            stats.update({
                "avg_reward": np.mean(recent_rewards),
                "std_reward": np.std(recent_rewards),
                "min_reward": np.min(recent_rewards),
                "max_reward": np.max(recent_rewards),
                "total_rewards": len(self.reward_history)
            })
        
        # System info
        stats.update(self.update_system_metrics())
        
        return stats
    
    def log_performance(self, prefix: str = "") -> None:
        """Log current performance statistics."""
        stats = self.get_performance_stats()
        
        log_msg = f"{prefix}Performance: "
        log_msg += f"FPS={stats.get('current_fps', 0):.1f}, "
        log_msg += f"Memory={stats.get('current_memory_mb', 0):.1f}MB, "
        log_msg += f"CPU={stats.get('current_cpu_percent', 0):.1f}%"
        
        if 'avg_reward' in stats:
            log_msg += f", Avg Reward={stats['avg_reward']:.3f}"
        
        self.logger.info(log_msg)
    
    def save_performance_log(self, filepath: str) -> None:
        """Save performance history to file."""
        try:
            log_data = {
                "fps_history": list(self.fps_history),
                "memory_history": list(self.memory_history),
                "cpu_history": list(self.cpu_history),
                "reward_history": list(self.reward_history),
                "timestamp": time.time()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            self.logger.info(f"Performance log saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance log: {e}")
    
    def reset(self) -> None:
        """Reset all performance history."""
        self.fps_history.clear()
        self.memory_history.clear()
        self.cpu_history.clear()
        self.reward_history.clear()
        self.frame_times.clear()
        self.start_time = time.time()
        self.logger.info("Performance monitor reset")
