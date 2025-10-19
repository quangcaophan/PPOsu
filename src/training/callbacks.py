"""
Custom callbacks for PPO training.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# Note: stable_baselines3.common.schedules doesn't exist in current version
# We'll implement learning rate scheduling manually

from ..core.logger import get_logger


class SongFinishedEvalCallback(EvalCallback):
    """
    Evaluation callback with cooldown to prevent too frequent evaluations.
    Only evaluates after songs are completed.
    """
    
    def __init__(
        self, 
        eval_env, 
        best_model_save_path: Optional[str] = None,
        n_eval_episodes: int = 2,
        min_songs_between_eval: int = 3,
        verbose: int = 1,
        **kwargs
    ):
        """
        Initialize song-finished evaluation callback.
        
        Args:
            eval_env: Evaluation environment
            best_model_save_path: Path to save best model
            n_eval_episodes: Number of episodes to evaluate
            min_songs_between_eval: Minimum songs between evaluations
            verbose: Verbosity level
        """
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            n_eval_episodes=n_eval_episodes,
            eval_freq=1,  # Check every step
            verbose=verbose,
            **kwargs
        )
        
        self.min_songs_between_eval = min_songs_between_eval
        self.songs_since_last_eval = 0
        self.total_songs_completed = 0
        self.logger = get_logger("eval_callback")
    
    def _on_step(self) -> bool:
        """Check if evaluation should be triggered."""
        # Check if an episode has ended
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            
            # Prefer explicit flag from the environment; fallback to results screen
            song_finished = bool(info.get("song_finished")) or info.get("game_state") == 7
            if song_finished:
                self.songs_since_last_eval += 1
                self.total_songs_completed += 1
                
                # Only evaluate every N songs
                if self.songs_since_last_eval >= self.min_songs_between_eval:
                    if self.verbose > 0:
                        self.logger.info(
                            f"Song #{self.total_songs_completed} finished! "
                            f"Triggering evaluation... (every {self.min_songs_between_eval} songs)"
                        )
                    
                    # Reset counter
                    self.songs_since_last_eval = 0
                    
                    # Run evaluation
                    original_n_calls = self.n_calls
                    self.n_calls = self.eval_freq
                    continue_training = super()._on_step()
                    
                    if self.verbose > 0:
                        self.logger.info("Evaluation complete! Resuming training.")
                    
                    # Signal to the training environment(s) that evaluation has completed
                    try:
                        # If vectorized, broadcast to all; otherwise, call directly
                        if hasattr(self.model, "get_env") and self.model.get_env() is not None:
                            vec_env = self.model.get_env()
                            # Try stable-baselines3 VecEnv API
                            if hasattr(vec_env, "env_method"):
                                vec_env.env_method("notify_evaluation_complete")
                            elif hasattr(vec_env, "get_attr"):
                                envs = vec_env.get_attr("envs", None)
                                for env in envs or []:
                                    if hasattr(env, "notify_evaluation_complete"):
                                        env.notify_evaluation_complete()
                        else:
                            # Fallback: best-effort, no-op if unavailable
                            pass
                    except Exception as e:
                        self.logger.error(f"Failed to send evaluation completion signal: {e}")
                    
                    # Restart a new beatmap in the evaluation environment
                    try:
                        if hasattr(self.eval_env, "reset"):
                            self.eval_env.reset()
                            if self.verbose > 0:
                                self.logger.info("Evaluation env reset: started a new beatmap.")
                    except Exception as e:
                        self.logger.error(f"Failed to restart beatmap after evaluation: {e}")
                    
                    self.n_calls = original_n_calls
                    return continue_training
                else:
                    if self.verbose > 0:
                        self.logger.info(
                            f"Song #{self.total_songs_completed} finished. "
                            f"Will evaluate in {self.min_songs_between_eval - self.songs_since_last_eval} more song(s)."
                        )
        
        return True


class CurriculumCallback(BaseCallback):
    """
    Gradually increase difficulty over time by adjusting reward parameters.
    """
    
    def __init__(
        self, 
        env, 
        initial_timesteps: int = 20000,
        verbose: int = 0
    ):
        """
        Initialize curriculum learning callback.
        
        Args:
            env: Training environment
            initial_timesteps: Timesteps before curriculum starts
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.initial_timesteps = initial_timesteps
        self.logger = get_logger("curriculum_callback")
    
    def _on_step(self) -> bool:
        """Update curriculum parameters."""
        # After initial timesteps, gradually increase difficulty
        if self.num_timesteps > self.initial_timesteps:
            progress = min(
                (self.num_timesteps - self.initial_timesteps) / 100000, 1.0
            )
            
            # Gradually increase idle penalty
            base_idle = -0.002
            target_idle = -0.01
            new_idle = base_idle + (target_idle - base_idle) * progress
            
            # Gradually increase miss penalty
            base_miss = -0.5
            target_miss = -2.0
            new_miss = base_miss + (target_miss - base_miss) * progress
            
            # Update environment reward parameters
            if hasattr(self.env, 'reward_calculator'):
                self.env.reward_calculator.reward_params.idle_penalty = new_idle
                self.env.reward_calculator.reward_params.miss_penalty = new_miss
            
            if self.verbose > 0 and self.num_timesteps % 10000 == 0:
                self.logger.info(
                    f"Curriculum Progress: {progress*100:.1f}% - "
                    f"Idle Penalty: {new_idle:.4f}, Miss Penalty: {new_miss:.2f}"
                )
        
        return True


class BehaviorMonitorCallback(BaseCallback):
    """
    Monitor agent's action distribution during training.
    """
    
    def __init__(
        self, 
        check_freq: int = 2000,
        verbose: int = 0
    ):
        """
        Initialize behavior monitoring callback.
        
        Args:
            check_freq: Frequency to check behavior
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_counts = {}
        self.logger = get_logger("behavior_monitor")
    
    def _on_step(self) -> bool:
        """Monitor action distribution."""
        action = self.locals.get('actions', [None])[0]
        if action is not None:
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        if self.n_calls % self.check_freq == 0:
            total = sum(self.action_counts.values())
            if total > 0:
                self.logger.info(f"Action Distribution (last {self.check_freq} steps):")
                
                # Count "no action" (action 0)
                no_action_pct = (self.action_counts.get(0, 0) / total) * 100
                self.logger.info(f"  No Action (0): {no_action_pct:.1f}%")
                
                # Count single key actions (1, 2, 4, 8)
                single_key_count = sum(self.action_counts.get(i, 0) for i in [1, 2, 4, 8])
                single_key_pct = (single_key_count / total) * 100
                self.logger.info(f"  Single Keys: {single_key_pct:.1f}%")
                
                # Count multi-key actions
                multi_key_count = total - self.action_counts.get(0, 0) - single_key_count
                multi_key_pct = (multi_key_count / total) * 100
                self.logger.info(f"  Multi Keys: {multi_key_pct:.1f}%")
                
                # Show top 5 most used actions
                top_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                self.logger.info(f"  Top 5 actions: {[(a, f'{(c/total)*100:.1f}%') for a, c in top_actions]}")
                
                if no_action_pct > 85:
                    self.logger.warning(
                        "⚠️ Agent is too passive! Consider increasing exploration."
                    )
                elif no_action_pct < 10:
                    self.logger.warning(
                        "⚠️ Agent is too active! May be spamming keys."
                    )
            
            self.action_counts.clear()
        
        return True


class LearningRateScheduler(BaseCallback):
    """
    Schedule learning rate during training.
    """
    
    def __init__(
        self,
        initial_lr: Optional[float] = None,
        final_lr: Optional[float] = None,
        verbose: int = 0
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate (defaults to optimizer LR at start)
            final_lr: Final learning rate (defaults to initial_lr)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps: Optional[int] = None
        self.logger = get_logger("lr_scheduler")

    def _on_training_start(self) -> None:
        """Initialize scheduler parameters once training starts."""
        # Determine total timesteps from the algorithm
        self.total_timesteps = getattr(self.model, "_total_timesteps", None)
        if self.total_timesteps is None or self.total_timesteps <= 0:
            # Fallback to avoid division by zero
            self.total_timesteps = 1

        # Infer starting LR from optimizer if not provided
        if hasattr(self.model, 'policy') and self.model.policy is not None:
            try:
                current_lr = self.model.policy.optimizer.param_groups[0]['lr']
                if self.initial_lr is None:
                    self.initial_lr = float(current_lr)
                if self.final_lr is None:
                    self.final_lr = float(current_lr)
            except Exception:
                # Use sane defaults if optimizer unavailable
                if self.initial_lr is None:
                    self.initial_lr = 3e-4
                if self.final_lr is None:
                    self.final_lr = self.initial_lr
        else:
            # No policy/optimizer yet
            if self.initial_lr is None:
                self.initial_lr = 3e-4
            if self.final_lr is None:
                self.final_lr = self.initial_lr

        if self.verbose > 0:
            self.logger.info(
                f"LR schedule initialized: start={self.initial_lr:.6f}, end={self.final_lr:.6f}, total_timesteps={self.total_timesteps}"
            )
    
    def _on_step(self) -> bool:
        """Update learning rate."""
        if not hasattr(self.model, 'policy') or self.model.policy is None:
            return True

        # Guard values
        total = max(int(self.total_timesteps or 1), 1)
        start_lr = float(self.initial_lr or 3e-4)
        end_lr = float(self.final_lr or start_lr)

        progress = min(self.num_timesteps / total, 1.0)
        new_lr = start_lr + (end_lr - start_lr) * progress
        
        # Update learning rate
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            self.logger.info(f"Learning rate: {new_lr:.6f}")
        
        return True


class PerformanceMonitorCallback(BaseCallback):
    """
    Monitor training performance and log statistics.
    """
    
    def __init__(
        self,
        check_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Initialize performance monitoring callback.
        
        Args:
            check_freq: Frequency to check performance
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.logger = get_logger("performance_monitor")
        self.reward_history = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Monitor performance metrics."""
        # Collect reward data
        if 'rewards' in self.locals:
            self.reward_history.extend(self.locals['rewards'])
        
        # Collect episode length data
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_lengths.append(info['episode']['l'])
        
        if self.n_calls % self.check_freq == 0:
            self._log_performance_stats()
        
        return True
    
    def _log_performance_stats(self):
        """Log performance statistics."""
        if not self.reward_history:
            return
        
        recent_rewards = self.reward_history[-1000:]  # Last 1000 rewards
        
        stats = {
            "avg_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "total_rewards": len(self.reward_history)
        }
        
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-100:]  # Last 100 episodes
            stats.update({
                "avg_episode_length": np.mean(recent_lengths),
                "total_episodes": len(self.episode_lengths)
            })
        
        self.logger.info(f"Performance Stats: {stats}")
        
        # Keep history manageable
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-5000:]
        if len(self.episode_lengths) > 1000:
            self.episode_lengths = self.episode_lengths[-500:]
