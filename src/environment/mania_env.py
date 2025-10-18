"""
Refactored osu!mania environment for PPO training.
Clean, modular design with separated concerns.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Tuple
import pydirectinput
import threading
from queue import Queue, Empty

from .constants import (
    KEY_MAPPINGS, FRAME_SIZE, VISUALIZATION_SIZE, FRAME_DELAY, 
    VIS_SIZE, GAME_STATE_PLAYING, NO_DATA_THRESHOLD
)
from .frame_processor import FrameProcessor
from .reward_calculator import RewardCalculator, GameState
from ..core.config_manager import AgentConfig
from ..core.logger import get_logger

# Import memory reader from original location
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from memory_reader import UltraFastMemoryReader as MemoryReader


class OsuManiaEnv(gym.Env):
    """
    Refactored osu!mania environment with clean separation of concerns.
    
    Features:
    - Modular frame processing
    - Separate reward calculation
    - Better error handling
    - Comprehensive logging
    - Type hints throughout
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self, 
        config: AgentConfig, 
        show_window: bool = False, 
        run_id: str = "default", 
        is_eval_env: bool = False
    ):
        """
        Initialize the osu!mania environment.
        
        Args:
            config: Agent configuration
            show_window: Whether to show visualization window
            run_id: Unique identifier for this run
            is_eval_env: Whether this is an evaluation environment
        """
        super().__init__()
        
        self.config = config
        self.show_window = show_window
        self.run_id = run_id
        self.is_eval_env = is_eval_env
        self.logger = get_logger(f"env_{run_id}")
        
        # Validate configuration
        if config.num_keys not in KEY_MAPPINGS:
            raise ValueError(f"Unsupported key mode: {config.num_keys}K")
        
        # Setup key mappings
        self.keys = KEY_MAPPINGS[config.num_keys]
        self.num_keys = config.num_keys
        
        # Initialize components
        self.frame_processor = FrameProcessor(config.play_area.__dict__)
        self.reward_calculator = RewardCalculator(config.reward_params)
        self.memory_reader = MemoryReader()
        
        # Gym spaces
        self.action_space = spaces.Discrete(2**self.num_keys)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(4, FRAME_SIZE, FRAME_SIZE), 
            dtype=np.float32
        )
        
        # State management
        self.last_four_frames = np.zeros((4, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        self.previous_keys_state = [False] * self.num_keys
        self.step_count = 0
        self.no_data_steps = 0
        self.user_quit = False
        
        # Game state
        self.current_game_state = GameState()
        
        # Start frame processing
        self.frame_processor.start()
        
        self.logger.info(f"Environment initialized for osu!mania {self.num_keys}K mode")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute (bitmask for key presses)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        step_start = time.time()
        self.step_count += 1
        
        # Convert action to key states
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # Execute action
        self._execute_action(action_combo)
        
        # Wait for frame update
        time.sleep(FRAME_DELAY)
        
        # Get new frame
        new_frame = self._get_latest_frame()
        self._update_frame_buffer(new_frame)
        
        # Get game state
        self._update_game_state()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.current_game_state, action, sum(action_combo)
        )
        
        # Check termination
        terminated = self._is_episode_ended()
        truncated = self.step_count >= self.config.max_steps
        
        # Visualization
        if self.show_window:
            self._render(action_combo, reward, step_start)
            if self._check_user_quit():
                self.user_quit = True
                terminated = True
        
        # Maintain FPS
        self._maintain_fps(step_start)
        
        # Prepare info
        info = self._prepare_info()
        
        return self.last_four_frames.copy(), reward, terminated, truncated, info
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Initial observation and info
        """
        super().reset(seed=seed)
        self.logger.info("Resetting environment... Starting new song!")
        
        # Release all keys
        self._release_all_keys()
        
        # Reset state
        self._reset_internal_state()
        
        # Handle evaluation environment auto-restart
        if self.is_eval_env:
            self._handle_eval_restart()
        
        # Wait for game to be ready
        self._wait_for_game_ready()
        
        # Initialize frame buffer
        self._initialize_frame_buffer()
        
        return self.last_four_frames.copy(), {}
    
    def close(self) -> None:
        """Clean up environment resources."""
        self.logger.info("Closing environment...")
        
        # Stop frame processing
        self.frame_processor.stop()
        
        # Release all keys
        self._release_all_keys()
        
        # Close visualization
        if self.show_window:
            cv2.destroyAllWindows()
        
        # Close memory reader
        if hasattr(self, "memory_reader"):
            self.memory_reader.close()
        
        self.logger.info("Environment closed")
    
    def _execute_action(self, action_combo: list) -> None:
        """Execute key actions safely."""
        try:
            for i, (key, current_state) in enumerate(zip(self.keys, action_combo)):
                previous_state = self.previous_keys_state[i]
                
                if current_state != previous_state:
                    if current_state:
                        pydirectinput.keyDown(key)
                    else:
                        pydirectinput.keyUp(key)
            
            self.previous_keys_state = action_combo.copy()
            
        except Exception as e:
            self.logger.error(f"Action execution error: {e}")
    
    def _get_latest_frame(self) -> np.ndarray:
        """Get the latest processed frame."""
        frame = self.frame_processor.get_frame()
        if frame is None:
            # Use last frame if no new frame available
            frame = self.last_four_frames[-1]
        return frame
    
    def _update_frame_buffer(self, new_frame: np.ndarray) -> None:
        """Update the frame buffer with new frame."""
        self.last_four_frames[:-1] = self.last_four_frames[1:]
        self.last_four_frames[-1] = new_frame
    
    def _update_game_state(self) -> None:
        """Update game state from memory reader."""
        try:
            game_data = self.memory_reader.get_game_state()
            
            if not game_data.get('fetch_successful', True):
                self.logger.warning("Communication with gosumemory lost")
                self.no_data_steps += 1
            else:
                self.no_data_steps = 0
            
            # Update current game state
            self.current_game_state.game_state = game_data.get('game_state', 0)
            self.current_game_state.score = game_data.get('score', 0)
            self.current_game_state.combo = game_data.get('combo', 0)
            self.current_game_state.accuracy = game_data.get('accuracy', 1.0)
            
            # Update hits
            self.current_game_state.hits = {
                k: v for k, v in game_data.items() 
                if k.startswith('hit') or k == 'miss'
            }
            
            # Auto-reset if no data for too long
            if self.no_data_steps > NO_DATA_THRESHOLD:
                self.logger.error("Auto-resetting due to memory reader failures")
                self.reset()
                self.no_data_steps = 0
                
        except Exception as e:
            self.logger.error(f"Game state update error: {e}")
    
    def _is_episode_ended(self) -> bool:
        """Check if episode should end."""
        return (self.current_game_state.game_state != GAME_STATE_PLAYING or 
                self.user_quit)
    
    def _check_user_quit(self) -> bool:
        """Check if user wants to quit."""
        return (cv2.waitKey(1) & 0xFF) == ord('q')
    
    def _render(self, action_combo: list, reward: float, step_start: float) -> None:
        """Render visualization window."""
        if not self.show_window:
            return
        
        # Create visualization frame
        vis_frame = self._create_visualization_frame(action_combo, reward, step_start)
        cv2.imshow(f'Osu! Mania AI - {self.num_keys}K', vis_frame)
    
    def _create_visualization_frame(
        self, 
        action_combo: list, 
        reward: float, 
        step_start: float
    ) -> np.ndarray:
        """Create visualization frame."""
        # Get latest frame
        frame = self.last_four_frames[-1]
        vis_frame = cv2.cvtColor(
            cv2.resize(frame, (VIS_SIZE, VIS_SIZE)), 
            cv2.COLOR_GRAY2BGR
        )
        
        # Draw key indicators
        key_width = VIS_SIZE // self.num_keys
        for i, pressed in enumerate(action_combo):
            x = i * key_width
            color = (0, 255, 0) if pressed else (0, 0, 255)
            cv2.rectangle(vis_frame, (x, 0), (x + key_width, 25), color, -1)
        
        # Draw info text
        y, h = 50, 25
        info_texts = [
            f"Reward: {reward:.2f}",
            f"Combo: {self.current_game_state.combo}",
            f"Acc: {self.current_game_state.accuracy*100:.1f}%",
            f"Miss: {self.current_game_state.hits.get('miss', 0)}",
            f"FPS: {1.0 / max(time.time() - step_start, 0.001):.1f}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(
                vis_frame, text, (10, y + i*h), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
        
        return vis_frame
    
    def _maintain_fps(self, step_start: float) -> None:
        """Maintain target FPS."""
        elapsed = time.time() - step_start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)
    
    def _prepare_info(self) -> Dict[str, Any]:
        """Prepare info dictionary."""
        return {
            'combo': self.current_game_state.combo,
            'score': self.current_game_state.score,
            'accuracy': self.current_game_state.accuracy,
            'game_state': self.current_game_state.game_state,
            'step_count': self.step_count,
            'reward_stats': self.reward_calculator.get_reward_stats(),
            **self.current_game_state.hits
        }
    
    def _release_all_keys(self) -> None:
        """Release all keys."""
        for key in self.keys:
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
    
    def _reset_internal_state(self) -> None:
        """Reset internal state variables."""
        self.user_quit = False
        self.step_count = 0
        self.no_data_steps = 0
        self.previous_keys_state = [False] * self.num_keys
        self.current_game_state = GameState()
        self.reward_calculator.reset()
    
    def _handle_eval_restart(self) -> None:
        """Handle automatic restart for evaluation environment."""
        self.logger.info("Evaluation environment: attempting to restart map")
        
        for attempt in range(3):
            time.sleep(3.0)
            pydirectinput.press('esc')
            time.sleep(5.0)
            pydirectinput.press('enter')
            time.sleep(1.0)
            pydirectinput.press('enter')
            
            # Check if game state is correct
            game_data = self.memory_reader.get_game_state()
            if game_data.get('game_state') == GAME_STATE_PLAYING:
                break
            
            self.logger.warning(
                f"Retry {attempt+1}: Game state is {game_data.get('game_state')}, "
                f"not {GAME_STATE_PLAYING} (Playing)"
            )
    
    def _wait_for_game_ready(self) -> None:
        """Wait for game to be in playing state."""
        time.sleep(3)
        
        max_wait = 15
        waited = 0
        while (self.current_game_state.game_state != GAME_STATE_PLAYING and 
               waited < max_wait):
            self._update_game_state()
            time.sleep(0.5)
            waited += 0.5
        
        if self.current_game_state.game_state != GAME_STATE_PLAYING:
            self.logger.warning(
                f"Game state is {self.current_game_state.game_state}, "
                f"not {GAME_STATE_PLAYING} (Playing)"
            )
    
    def _initialize_frame_buffer(self) -> None:
        """Initialize frame buffer with current frames."""
        for i in range(4):
            frame = self._get_latest_frame()
            self.last_four_frames[i] = frame
            time.sleep(0.05)
