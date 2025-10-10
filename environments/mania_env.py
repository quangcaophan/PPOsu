import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import time
from collections import deque
import json
from typing import Dict, Any
import pydirectinput

from environments.constants import KEY_MAPPINGS, FRAME_SIZE, VISUALIZATION_SIZE, FRAME_DELAY
from performance_profiler import time_operation
from memory_reader import MemoryReader
from tracing import JSONTracer


class OsuManiaEnv(gym.Env):
    """
    Optimized osu!mania Environment và Performance Optimizations
    """

    def __init__(self, config_path: str, show_window=True, run_id: str = "default", log_dir: str = "logs"):
        super(OsuManiaEnv, self).__init__()
        
        # Load config
        self.config = self._load_config(config_path)
        if not self.config:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Extract config values
        self.play_area = self.config.get('play_area')
        self.num_keys = self.config.get('num_keys', 4)
        self.max_steps = self.config.get('max_steps', 15000)
        
        # Load reward parameters from config
        reward_params = self.config.get('reward_params', {})
        self.miss_penalty = reward_params.get('miss_penalty', -5.0)
        self.key_spam_penalty = reward_params.get('key_spam_penalty', -0.01)
        self.idle_penalty = reward_params.get('idle_penalty', -0.05)
        self.menu_key_penalty = reward_params.get('menu_key_penalty', -0.5)
        self.menu_idle_reward = reward_params.get('menu_idle_reward', 0.1)

        # Setup key mappings
        self.keys = KEY_MAPPINGS.get(self.num_keys)
        if not self.keys:
            raise ValueError(f"Unsupported key mode: {self.num_keys}K")

        # Initialize components
        self.sct = mss.mss()
        self.show_window = show_window
        self.memory_reader = MemoryReader()
        
        # Gym spaces
        self.action_space = spaces.Discrete(2**self.num_keys)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        
        # State variables
        self.last_four_frames = np.zeros((4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.previous_keys_state = [False] * self.num_keys
        self.step_count = 0
        
        # Game state
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.prev_score = 0, 0
        self.last_accuracy, self.prev_accuracy = 1.0, 1.0
        self.last_hits = {}
        self.prev_hits = {}
        self.game_state = 0
        
        # Control variables
        self.user_quit = False
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)

        # Debugging
        if run_id and log_dir:
            self.tracer = JSONTracer(log_dir=log_dir, run_id=run_id)
        
        self.log(f"Environment initialized for osu!mania {self.num_keys}K mode")

    def log(self, message, level="INFO"):
        """Simple logging"""
        prefix = {"INFO": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
        print(f"{prefix} {message}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f: 
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _get_state(self):
        """Optimized state capture"""
        with time_operation('env_get_state'):
            try:
                frame_start = time.time()
                sct_img = self.sct.grab(self.play_area)
                img = np.array(sct_img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
                
                # Performance tracking
                self.frame_times.append(time.time() - frame_start)
                return resized
            except Exception:
                return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

    def _calculate_reward(self):
        """
        Calculates reward based on the change in hit counts.
        This provides a much more direct signal for the agent to learn from.
        """
        reward = 0.0
        num_keys_pressed = sum(self.previous_keys_state)
        is_gameplay_active = self.game_state == 2 # 2 is the state for "Playing"

        if is_gameplay_active and self.prev_hits:
            # Calculate the difference in hits from the last step
            hit_diffs = {k: self.last_hits.get(k, 0) - self.prev_hits.get(k, 0) for k in self.last_hits}

            # Assign rewards based on new hits
            reward += hit_diffs.get('hit_geki', 0) * 3.0  # Highest reward for perfect
            reward += hit_diffs.get('hit_300', 0) * 2.0
            reward += hit_diffs.get('hit_100', 0) * 0.5
            reward += hit_diffs.get('hit_50', 0) * -0.5 # Penalty for 50s
            reward += hit_diffs.get('miss', 0) * -5.0   # Strong penalty for misses

            # Combo reward
            if self.last_combo > self.prev_combo:
                reward += (self.last_combo * 0.01)
            
            # Penalty for breaking combo
            if self.last_combo == 0 and self.prev_combo > 10:
                reward += self.miss_penalty

            # Key spam penalty
            reward += self.key_spam_penalty * num_keys_pressed if num_keys_pressed > 0 else self.idle_penalty
        else:
            # Menu state
            reward += self.menu_key_penalty if num_keys_pressed > 0 else self.menu_idle_reward

        # Update previous values for the next step
        self.prev_combo = self.last_combo
        self.prev_score = self.last_score
        self.prev_accuracy = self.last_accuracy
        self.prev_hits = self.last_hits.copy()
        
        return reward

    def _is_game_ended(self) -> bool:
        """
        Check if game has ended based on game state from memory reader.
        State 7 is the results screen.
        State 5 is the menu screen.
        """
        return self.game_state != 2 or self.user_quit

    def _execute_action_safely(self, action_combo):
        """Execute key actions with error handling"""
        for i in range(self.num_keys):
            if self.previous_keys_state[i] != action_combo[i]:
                try:
                    if action_combo[i]: 
                        pydirectinput.keyDown(self.keys[i])
                    else: 
                        pydirectinput.keyUp(self.keys[i])
                except Exception:
                    pass  # Ignore key errors to prevent crashes
        self.previous_keys_state = action_combo

    def step(self, action):
        """Main step function với full optimization"""
        step_start = time.time()
        self.step_count += 1
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # 1. Execute action FIRST (highest priority)
        self._execute_action_safely(action_combo)
        
        # 2. Capture frame
        new_frame = self._get_state()
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0)
        self.last_four_frames[-1] = new_frame
        
        # 3. Get game state from RAM
        game_state = self.memory_reader.get_game_state()
        self.game_state = game_state.get('game_state')
        self.last_score = game_state.get('score')
        self.last_combo = game_state.get('combo')
        self.last_accuracy = game_state.get('accuracy')
        self.last_hits = {k: v for k, v in game_state.items() if 'hit' in k or k == 'miss'}
        
        # 4. Calculate reward
        reward = self._calculate_reward()
        
        # 5. Check termination
        terminated = self._is_game_ended()
        truncated = self.step_count >= self.max_steps
        
        # 7. Visualization
        if self.show_window:
            self._show_visualization(new_frame, action_combo, reward, step_start)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.user_quit = True

        # 8. Maintain FPS
        elapsed = time.time() - step_start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

        info = {
            'combo': self.last_combo if self.last_combo is not None else self.prev_combo,
            'score': self.last_score if self.last_score is not None else self.prev_score,
            'accuracy': self.last_accuracy if self.last_accuracy is not None else self.prev_accuracy,
            'fps': 1.0 / max(time.time() - step_start, 0.001),
            **self.last_hits
        }
        
        # 9. JSON Tracing
        if hasattr(self, 'tracer'):
            trace_data = {
                'step': self.step_count,
                'timestamp': time.time(),
                'action': int(action),
                'action_combo': action_combo,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'game_state': self.game_state,
                **info
            }
            self.tracer.log_step(trace_data)

        return self.last_four_frames.copy(), reward, terminated, truncated, info

    def _show_visualization(self, frame, action_combo, reward, step_start):
        """Enhanced visualization"""
        vis_frame = cv2.cvtColor(cv2.resize(frame, (VISUALIZATION_SIZE, VISUALIZATION_SIZE)), cv2.COLOR_GRAY2BGR)
        key_width = VISUALIZATION_SIZE // self.num_keys
        
        # Key indicators
        for i in range(self.num_keys):
            x = i * key_width
            color = (0, 255, 0) if action_combo[i] else (0, 0, 255)
            cv2.rectangle(vis_frame, (x, 0), (x + key_width, 25), color, -1)
        
        # Game info
        y, h = 50, 25
        current_fps = 1.0 / max(time.time() - step_start, 0.001)
        avg_fps = len(self.frame_times) / max(sum(self.frame_times), 0.001) if self.frame_times else 0
        
        info_texts = [
            f"Reward: {reward:.2f}",
            f"Combo: {self.last_combo}",
            f"Acc: {self.last_accuracy*100:.1f}%",
            f"Geki: {self.last_hits.get('hit_geki', 0)}",
            f"300: {self.last_hits.get('hit_300', 0)}",
            f"100: {self.last_hits.get('hit_100', 0)}",
            f"50: {self.last_hits.get('hit_50', 0)}",
            f"Miss: {self.last_hits.get('miss', 0)}"
        ]
        
        for i, text in enumerate(info_texts):
            color = (255, 255, 0) if i < 4 else (255, 0, 255)
            cv2.putText(vis_frame, text, (10, y + i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(vis_frame, "Press 'q' to quit", (10, VISUALIZATION_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow(f'Osu! Mania AI - {self.num_keys}K', vis_frame)

    def reset(self, seed=None, options=None):
        """Reset environment state"""
        super().reset(seed=seed)
        self.log("Resetting environment... Start new song!")

        # Release keys
        for key in self.keys: 
            try: 
                pydirectinput.keyUp(key)
            except: 
                pass

        # Reset state
        self.user_quit = False
        self.step_count = 0
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.prev_score = 0, 0
        self.last_accuracy, self.prev_accuracy = 1.0, 1.0
        self.last_hits, self.prev_hits = {}, {}
        self.game_state = 0
        self.frame_times.clear()

        # Start new trace file for the episode
        if hasattr(self, 'tracer'):
            self.tracer.start_episode()

        time.sleep(3)
        
        # Initialize frames
        for i in range(4):
            frame = self._get_state()
            self.last_four_frames[i] = frame
            time.sleep(0.05)
            
        return self.last_four_frames.copy(), {}

    def close(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        
        # Release keys
        for key in self.keys: 
            try: 
                pydirectinput.keyUp(key)
            except: 
                pass
                
        # Close tracer
        if hasattr(self, 'tracer'):
            self.tracer.close()

        self.log("Environment closed")
