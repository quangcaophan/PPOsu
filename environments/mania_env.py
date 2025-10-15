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
import threading
from queue import Queue, Empty # Import 'Empty' để bắt lỗi cụ thể

from environments.constants import (
    KEY_MAPPINGS, FRAME_SIZE, VISUALIZATION_SIZE, FRAME_DELAY, 
    VIS_SIZE, DEFAULT_REWARD_PARAMS
)
from memory_reader import UltraFastMemoryReader as MemoryReader

class OsuManiaEnv(gym.Env):
    """
    Optimized osu!mania Environment with Performance Optimizations.
    """

    def __init__(self, config_path: str, show_window=False, run_id: str = "default", is_eval_env=False):
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
        self.reward_values = {**DEFAULT_REWARD_PARAMS, **reward_params}

        # Setup key mappings
        self.keys = KEY_MAPPINGS.get(self.num_keys)
        if not self.keys:
            raise ValueError(f"Unsupported key mode: {self.num_keys}K")

        # Initialize components
        self.sct = mss.mss()
        self.show_window = show_window
        self.memory_reader = MemoryReader()
        self.is_eval_env = is_eval_env
        
        # Gym spaces
        self.action_space = spaces.Discrete(2**self.num_keys)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        
        # State variables
        self.last_four_frames = np.zeros((4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.previous_keys_state = [False] * self.num_keys
        self.step_count = 0
        self.no_data_steps = 0
        self.NO_DATA_THRESHOLD = 30  # ~0.5 seconds of no data
        
        # Game state trackers
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.prev_score = 0, 0
        self.last_accuracy, self.prev_accuracy = 1.0, 1.0
        self.last_hits, self.prev_hits = {}, {}
        self.game_state = 0
        
        # Control variables
        self.user_quit = False
        
        # Thread-safe frame communication
        self.frame_queue = Queue(maxsize=2)
        self.frame_intervals = deque(maxlen=120)
        self.crop_only = (
            self.play_area.get('width', FRAME_SIZE) == FRAME_SIZE and
            self.play_area.get('height', FRAME_SIZE) == FRAME_SIZE
        )
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.log(f"Environment initialized for osu!mania {self.num_keys}K mode")

    def log(self, message, level="INFO"):
        prefix = {"INFO": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
        print(f"{prefix} {message}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f: 
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _capture_loop(self):
        """Background thread to capture screen frames and put them in a queue with stable FPS."""
        try:
            sct_thread = mss.mss(with_cursor=False)
            print("✅ Capture thread started (using Queue)")
            next_frame_time = time.time()
            prev_time = time.time()
            while True:
                try:
                    sct_img = sct_thread.grab(self.play_area)
                    img = np.array(sct_img)
                    if self.crop_only:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    else:
                        resized = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
                    gray = cv2.GaussianBlur(gray, (3,3), 0)
                    gray = cv2.equalizeHist(gray)
                    norm_gray = gray.astype(np.float32) / 255.0
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                    self.frame_queue.put(norm_gray)
                    interval = (time.time() - prev_time) * 1000
                    self.frame_intervals.append(interval)
                    prev_time = time.time()
                    next_frame_time += 1/60
                    delay = next_frame_time - time.time()
                    if delay > 0:
                        time.sleep(delay)
                except Exception as e:
                    print(f"❌ Capture error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Capture thread initialization failed: {e}")

    def _get_state(self):
        """Pulls the latest normalized frame from the thread-safe queue."""
        try:
            frame = self.frame_queue.get_nowait()
        except Empty:
            frame = self.last_four_frames[-1]
        except Exception as e:
            print(f"❌ Frame queue error: {e}")
            frame = self.last_four_frames[-1]
        return frame.copy()

    def _calculate_reward(self):
            """
            Hàm phần thưởng được cải tiến với các tín hiệu học hỏi dày đặc và có ý nghĩa hơn.
            Bao gồm: Chi phí hành động, Thưởng cột mốc combo, và Thưởng theo thay đổi độ chính xác.
            """
            reward_values = self.reward_values            
            reward = reward_values.get("living_penalty", -0.01)
            num_keys_pressed = sum(self.previous_keys_state)
            action_cost = reward_values.get("action_cost_penalty", -0.005) * num_keys_pressed
            reward += action_cost
            is_gameplay_active = self.game_state == 2
            if is_gameplay_active:
                if self.prev_hits:
                    hit_diffs = {k: self.last_hits.get(k, 0) - self.prev_hits.get(k, 0) for k in self.last_hits}
                    
                    reward += hit_diffs.get('hit_geki', 0) * reward_values.get("hit_geki_reward", 5.0)
                    reward += hit_diffs.get('hit_300', 0) * reward_values.get("hit_300_reward", 3.0)
                    reward += hit_diffs.get('hit_100', 0) * reward_values.get("hit_100_reward", 1.0)
                    reward += hit_diffs.get('hit_50', 0) * reward_values.get("hit_50_penalty", -0.5)
                    reward += hit_diffs.get('miss', 0) * reward_values.get("miss_penalty", -1.0)

                if self.last_combo > self.prev_combo:
                    combo_increase = self.last_combo - self.prev_combo
                    reward += combo_increase * reward_values.get("combo_increase_reward", 0.1)
                    
                    if self.prev_combo < 50 and self.last_combo >= 50:
                        reward += reward_values.get("combo_milestone_50", 10.0)
                    if self.prev_combo < 100 and self.last_combo >= 100:
                        reward += reward_values.get("combo_milestone_100", 20.0)
                    if self.prev_combo < 200 and self.last_combo >= 200:
                        reward += reward_values.get("combo_milestone_200", 40.0)
                
                if self.last_combo == 0 and self.prev_combo > 5:
                    reward += reward_values.get("combo_break_penalty", -0.5)

                if self.last_accuracy is not None and self.prev_accuracy is not None:
                    accuracy_change = self.last_accuracy - self.prev_accuracy
                    accuracy_multiplier = reward_values.get("accuracy_change_multiplier", 100.0)
                    reward += accuracy_change * accuracy_multiplier

                if num_keys_pressed == 0:
                    reward += reward_values.get("idle_penalty", -0.02)
            
            self.prev_combo = self.last_combo
            self.prev_score = self.last_score
            self.prev_accuracy = self.last_accuracy
            self.prev_hits = self.last_hits.copy()
            
            return reward

    def _is_game_ended(self) -> bool:
        return self.game_state != 2 or self.user_quit

    def _execute_action_safely(self, action_combo):
        """
        Executes key actions sequentially and efficiently without creating new threads.
        """
        try:
            for i in range(self.num_keys):
                key = self.keys[i]
                current_state = action_combo[i]
                previous_state = self.previous_keys_state[i]
                if current_state != previous_state:
                    try:
                        if current_state:
                            pydirectinput.keyDown(key)
                        else:
                            pydirectinput.keyUp(key)
                    except Exception as e:
                        print(f"❌ Key action error for {key}: {e}")
            self.previous_keys_state = action_combo
        except Exception as e:
            print(f"❌ Action execution error: {e}")

    def step(self, action):
        step_start = time.time()
        self.step_count += 1
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # 1. Execute action
        self._execute_action_safely(action_combo)
        
        # --- State Consistency: Wait for next frame after action ---
        time.sleep(1/60)  # Wait for one frame tick to ensure new frame after action
        
        # 2. Capture frame
        new_frame = self._get_state()
        # --- Rolling buffer optimization ---
        self.last_four_frames[:-1] = self.last_four_frames[1:]
        self.last_four_frames[-1] = new_frame
        
        # 3. Get game state from RAM
        game_state_data = self.memory_reader.get_game_state()
        if not game_state_data.get('fetch_successful', True):
            self.log("Communication with gosumemory/tosu might be lost.", "WARNING")
            self.no_data_steps += 1
        else:
            self.no_data_steps = 0

        self.game_state = game_state_data.get('game_state')
        self.last_score = game_state_data.get('score')
        self.last_combo = game_state_data.get('combo')
        self.last_accuracy = game_state_data.get('accuracy')
        self.last_hits = {k: v for k, v in game_state_data.items() if 'hit' in k or k == 'miss'}

        # Auto-reset if memory reader fails repeatedly
        if self.no_data_steps > self.NO_DATA_THRESHOLD:
            self.log("Auto-resetting due to repeated memory reader failures.", "ERROR")
            self.reset()
            self.no_data_steps = 0

        # 4. Calculate reward
        reward = self._calculate_reward()
        
        # 5. Check termination
        terminated = self._is_game_ended()
        truncated = self.step_count >= self.max_steps
        
        # 6. Visualization
        if self.show_window:
            self._show_visualization(new_frame, action_combo, reward, step_start)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.user_quit = True
        
        # 7. Maintain FPS
        elapsed = time.time() - step_start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)

        info = {
            'combo': self.last_combo,
            'score': self.last_score,
            'accuracy': self.last_accuracy,
            'fps': 1.0 / max(time.time() - step_start, 0.001),
            'game_state': self.game_state,
            **self.last_hits
        }

        # --- Buffer is always normalized, so just return copy ---
        obs = self.last_four_frames.copy()
        return obs, reward, terminated, truncated, info

    def _show_visualization(self, frame, action_combo, reward, step_start):
        vis_frame = cv2.cvtColor(cv2.resize(frame, (VIS_SIZE, VIS_SIZE)), cv2.COLOR_GRAY2BGR)
        key_width = VIS_SIZE // self.num_keys
        
        for i in range(self.num_keys):
            x = i * key_width
            color = (0, 255, 0) if action_combo[i] else (0, 0, 255)
            cv2.rectangle(vis_frame, (x, 0), (x + key_width, 25), color, -1)
        
        y, h = 50, 25
        info_texts = [
            f"Reward: {reward:.2f}",
            f"Combo: {self.last_combo}",
            f"Acc: {self.last_accuracy*100:.1f}%",
            f"Miss: {self.last_hits.get('miss', 0)}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(vis_frame, text, (10, y + i*h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow(f'Osu! Mania AI - {self.num_keys}K', vis_frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.log("Resetting environment... Start new song!")

        for key in self.keys: 
            try: 
                pydirectinput.keyUp(key)
            except: 
                pass
        
        # Add retry logic for evaluation environment
        if self.is_eval_env:
            self.log("Evaluation environment: attempting to restart map automatically.")
            for attempt in range(3):
                time.sleep(3.0)
                pydirectinput.press('esc')
                time.sleep(5.0)
                pydirectinput.press('enter')
                time.sleep(1.0)
                pydirectinput.press('enter')
                # Check if game state is correct
                game_state_data = self.memory_reader.get_game_state()
                self.game_state = game_state_data.get('game_state')
                if self.game_state == 2:
                    break
                self.log(f"Retry {attempt+1}: Game state is {self.game_state}, not 2 (Playing)", "WARNING")

        self.user_quit = False
        self.step_count = 0
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.prev_score = 0, 0
        self.last_accuracy, self.prev_accuracy = 1.0, 1.0
        self.last_hits, self.prev_hits = {}, {}
        self.game_state = 0
        
        time.sleep(3)

        max_wait = 15
        waited = 0
        while self.game_state != 2 and waited < max_wait:
            game_state_data = self.memory_reader.get_game_state()
            self.game_state = game_state_data.get('game_state')
            time.sleep(0.5)
            waited += 0.5
        
        if self.game_state != 2:
            self.log(f"WARNING: Game state is {self.game_state}, not 2 (Playing)", "WARNING")
        
        for i in range(4):
            frame = self._get_state()
            self.last_four_frames[i] = frame
            time.sleep(0.05)
            
        return self.last_four_frames.copy(), {}

    def close(self):
        cv2.destroyAllWindows()
        for key in self.keys: 
            try: 
                pydirectinput.keyUp(key)
            except: 
                pass
        self.log("Environment closed")