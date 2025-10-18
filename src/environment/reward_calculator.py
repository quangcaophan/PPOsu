"""
Reward calculation system for the osu!mania environment.
Handles all reward logic and state tracking.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..core.config_manager import RewardParams


@dataclass
class GameState:
    """Current game state for reward calculation."""
    combo: int = 0
    score: int = 0
    accuracy: float = 1.0
    hits: Dict[str, int] = None
    game_state: int = 0
    
    def __post_init__(self):
        if self.hits is None:
            self.hits = {
                'hit_geki': 0,
                'hit_300': 0,
                'hit_100': 0,
                'hit_50': 0,
                'miss': 0
            }


class RewardCalculator:
    """Calculates rewards based on game state and performance."""
    
    def __init__(self, reward_params: RewardParams):
        """
        Initialize reward calculator.
        
        Args:
            reward_params: Reward function parameters
        """
        self.reward_params = reward_params
        self.previous_state = GameState()
        self.current_state = GameState()
        
        # Performance tracking
        self.total_reward = 0.0
        self.reward_history = []
    
    def calculate_reward(
        self,
        current_state: GameState,
        action_taken: int,
        num_keys_pressed: int
    ) -> float:
        """
        Calculate reward based on current state and action.
        
        Args:
            current_state: Current game state
            action_taken: Action taken by agent
            num_keys_pressed: Number of keys pressed in action
            
        Returns:
            Calculated reward value
        """
        # Update states
        self.previous_state = self.current_state
        self.current_state = current_state
        
        # Calculate base reward
        reward = self._calculate_base_reward(num_keys_pressed)
        
        # Add gameplay rewards if playing
        if current_state.game_state == 2:  # Playing state
            reward += self._calculate_hit_rewards()
            reward += self._calculate_combo_rewards()
            reward += self._calculate_accuracy_rewards()
            reward += self._calculate_exploration_bonus(num_keys_pressed)
            reward += self._calculate_idle_penalty(num_keys_pressed)
        
        # Track total reward
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Keep history manageable
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]
        
        return reward
    
    def _calculate_base_reward(self, num_keys_pressed: int) -> float:
        """Calculate base reward components."""
        reward = self.reward_params.living_penalty
        
        # Action cost penalty
        action_cost = self.reward_params.action_cost_penalty * num_keys_pressed
        reward += action_cost
        
        return reward
    
    def _calculate_hit_rewards(self) -> float:
        """Calculate rewards for hit accuracy."""
        if not self.previous_state.hits:
            return 0.0
        
        reward = 0.0
        hit_diffs = {
            k: self.current_state.hits.get(k, 0) - self.previous_state.hits.get(k, 0)
            for k in self.current_state.hits
        }
        
        # Hit rewards
        reward += hit_diffs.get('hit_geki', 0) * self.reward_params.hit_geki_reward
        reward += hit_diffs.get('hit_300', 0) * self.reward_params.hit_300_reward
        reward += hit_diffs.get('hit_100', 0) * self.reward_params.hit_100_reward
        reward += hit_diffs.get('hit_50', 0) * self.reward_params.hit_50_penalty
        reward += hit_diffs.get('miss', 0) * self.reward_params.miss_penalty
        
        return reward
    
    def _calculate_combo_rewards(self) -> float:
        """Calculate rewards for combo building."""
        reward = 0.0
        
        # Combo increase reward
        if self.current_state.combo > self.previous_state.combo:
            combo_increase = self.current_state.combo - self.previous_state.combo
            reward += combo_increase * self.reward_params.combo_increase_reward
            
            # Milestone rewards
            reward += self._calculate_milestone_rewards()
        
        # Combo break penalty
        elif self.current_state.combo == 0 and self.previous_state.combo > 5:
            reward += self.reward_params.combo_break_penalty
        
        return reward
    
    def _calculate_milestone_rewards(self) -> float:
        """Calculate milestone rewards for combo achievements."""
        reward = 0.0
        prev_combo = self.previous_state.combo
        current_combo = self.current_state.combo
        
        # Early encouragement
        if prev_combo < 10 and current_combo >= 10:
            reward += 2.0
        if prev_combo < 25 and current_combo >= 25:
            reward += 3.0
        
        # Major milestones
        if prev_combo < 50 and current_combo >= 50:
            reward += self.reward_params.combo_milestone_50
        if prev_combo < 100 and current_combo >= 100:
            reward += self.reward_params.combo_milestone_100
        if prev_combo < 200 and current_combo >= 200:
            reward += self.reward_params.combo_milestone_200
        
        return reward
    
    def _calculate_accuracy_rewards(self) -> float:
        """Calculate rewards for accuracy improvements."""
        if (self.current_state.accuracy is None or 
            self.previous_state.accuracy is None):
            return 0.0
        
        accuracy_change = self.current_state.accuracy - self.previous_state.accuracy
        return accuracy_change * self.reward_params.accuracy_change_multiplier
    
    def _calculate_exploration_bonus(self, num_keys_pressed: int) -> float:
        """Calculate exploration bonus to encourage trying actions."""
        if num_keys_pressed > 0 and self.current_state.combo < 20:
            return 0.01  # Small bonus for trying actions early
        return 0.0
    
    def _calculate_idle_penalty(self, num_keys_pressed: int) -> float:
        """Calculate idle penalty for not taking actions."""
        if num_keys_pressed == 0:
            # Only penalize after learning basics
            if self.current_state.combo > 10:
                return self.reward_params.idle_penalty
            else:
                return self.reward_params.idle_penalty * 0.5  # Lighter penalty early
        return 0.0
    
    def get_reward_stats(self) -> Dict[str, Any]:
        """Get reward calculation statistics."""
        if not self.reward_history:
            return {}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 rewards
        
        return {
            "total_reward": self.total_reward,
            "avg_reward": np.mean(recent_rewards),
            "std_reward": np.std(recent_rewards),
            "min_reward": np.min(recent_rewards),
            "max_reward": np.max(recent_rewards),
            "reward_count": len(self.reward_history)
        }
    
    def reset(self) -> None:
        """Reset reward calculator state."""
        self.previous_state = GameState()
        self.current_state = GameState()
        self.total_reward = 0.0
        self.reward_history = []
    
    def update_reward_params(self, new_params: RewardParams) -> None:
        """Update reward parameters."""
        self.reward_params = new_params
