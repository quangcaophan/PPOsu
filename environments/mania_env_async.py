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

from environments.constants import *
from performance_profiler import time_operation
from memory_reader import MemoryReader
from tracing import JSONTracer


class OsuManiaEnv(gym.Env):
    """
    Optimized osu!mania Environment với Async OCR và Performance Optimizations
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
        self.frame_buffer = deque(maxlen=10)
        self.step_count = 0
        self.max_steps = MAX_STEPS_DEFAULT
        
        # Game state
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.prev_score = 0, 0
        self.last_accuracy, self.prev_accuracy = 1.0, 1.0
        
        # Control variables
        self.last_activity_time = time.time()
        self.user_quit = False
        self.activity_score = 0.0
        self.result_template = None
        self.game_ended_frames = 0
        
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

    def load_result_template(self, template_path: str):
        """Load result screen template for game end detection"""
        try:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.result_template = cv2.resize(template, (FRAME_SIZE, FRAME_SIZE))
                self.log(f"Result template loaded from {template_path}")
        except Exception as e:
            self.log(f"Error loading template: {e}", "ERROR")

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
        """Optimized reward calculation"""
        reward = 0.0
        num_keys_pressed = sum(self.previous_keys_state)
        is_gameplay_active = self.activity_score > ACTIVITY_THRESHOLD

        if is_gameplay_active:
            # Miss penalty
            if self.last_combo == 0 and self.prev_combo > 10:
                reward += MISS_PENALTY

            # Hit reward
            score_diff = self.last_score - self.prev_score
            accuracy_diff = self.last_accuracy - self.prev_accuracy

            if score_diff > 0:
                if accuracy_diff >= -0.0001:
                    hit_reward = (0.5 + self.last_combo * 0.01) * self.last_accuracy
                    reward += hit_reward
                else:
                    reward += 0.1 
                    reward -= abs(accuracy_diff) * 2.0

            # Key spam penalty
            reward += KEY_SPAM_PENALTY * num_keys_pressed if num_keys_pressed > 0 else IDLE_PENALTY
        else:
            # Menu state
            reward += MENU_KEY_PENALTY if num_keys_pressed > 0 else MENU_IDLE_REWARD

        # Update previous values
        self.prev_combo = self.last_combo
        self.prev_score = self.last_score  
        self.prev_accuracy = self.last_accuracy
        
        return reward

    def _detect_game_activity(self, current_frame):
        """Detect game activity for state management"""
        if len(self.frame_buffer) < 2: 
            return 0.0
        diff = cv2.absdiff(current_frame, self.frame_buffer[-1])
        activity_score = np.sum(diff > 25) / (FRAME_SIZE * FRAME_SIZE)
        if activity_score > ACTIVITY_THRESHOLD: 
            self.last_activity_time = time.time()
        return activity_score

    def _is_game_ended(self, current_frame) -> bool:
        """Check if game has ended"""
        # Template matching
        if self.result_template is not None:
            try:
                res = cv2.matchTemplate(current_frame, self.result_template, cv2.TM_CCOEFF_NORMED)
                if np.max(res) > 0.8:
                    self.game_ended_frames += 1
                    if self.game_ended_frames > 5: 
                        return True
                else: 
                    self.game_ended_frames = 0
            except cv2.error: 
                pass
        
        # User quit or timeout
        if self.user_quit: 
            return True
        if time.time() - self.last_activity_time > GAME_END_TIMEOUT and self.last_combo == 0: 
            return True
        return False

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
        
        # 3. Update activity
        self.activity_score = self._detect_game_activity(new_frame)
        self.frame_buffer.append(new_frame)
        
        # 4. Get game state from RAM
        self.last_score, self.last_combo, self.last_accuracy = self.memory_reader.get_game_state()
        
        # 5. Calculate reward
        reward = self._calculate_reward()
        
        # 6. Check termination
        terminated = self._is_game_ended(new_frame)
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
            'fps': 1.0 / max(time.time() - step_start, 0.001)
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
                'activity_score': self.activity_score,
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
            f"Score: {self.last_score}",
            f"Acc: {self.last_accuracy*100:.1f}%",
            f"FPS: {current_fps:.1f}",
            f"Avg: {avg_fps:.1f}",
            f"Mode: {self.num_keys}K"
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
        self.last_activity_time = time.time()
        self.frame_buffer.clear()
        self.game_ended_frames = 0
        self.frame_times.clear()

        # Start new trace file for the episode
        if hasattr(self, 'tracer'):
            self.tracer.start_episode()

        time.sleep(3)
        
        # Initialize frames
        for i in range(4):
            frame = self._get_state()
            self.last_four_frames[i] = frame
            self.frame_buffer.append(frame)
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
