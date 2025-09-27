import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
from collections import deque
import easyocr
import json
from typing import Dict, Any
import re

KEY_MAPPINGS = {
    4: ['d', 'f', 'j', 'k'],
    5: ['d', 'f', 'space', 'j', 'k'],
    6: ['s', 'd', 'f', 'j', 'k', 'l'],
    7: ['s', 'd', 'f', 'space', 'j', 'k', 'l']
}

FRAME_SIZE = 64
VISUALIZATION_SIZE = 420
TARGET_FPS = 60
FRAME_DELAY = 1.0 / TARGET_FPS

class OsuManiaEnv(gym.Env):
    def __init__(self, config_path: str, show_window=True):
        super(OsuManiaEnv, self).__init__()
        
        # 1. Tải toàn bộ cấu hình từ file được chỉ định
        self.config = self._load_config(config_path)
        if not self.config:
            raise FileNotFoundError(f"Configuration file not found or is invalid: {config_path}")

        # 2. Lấy các thông số từ file config
        self.play_area = self.config.get('play_area')
        self.combo_area = self.config.get('combo_area')
        self.score_area = self.config.get('score_area')
        self.accuracy_area = self.config.get('accuracy_area')
        self.num_keys = self.config.get('num_keys', 4) # Mặc định là 4K nếu không có

        # 3. Chọn bộ phím phù hợp dựa trên num_keys
        self.keys = KEY_MAPPINGS.get(self.num_keys)
        if not self.keys:
            raise ValueError(f"Key mapping for {self.num_keys}K mode is not defined in KEY_MAPPINGS.")

        self.sct = mss.mss()
        self.show_window = show_window
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
        # 4. Action space và các biến khác sẽ tự động điều chỉnh theo num_keys
        self.action_space = spaces.Discrete(2**self.num_keys)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        
        self.last_four_frames = np.zeros((4, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
        self.previous_keys_state = [False] * self.num_keys
        self.frame_buffer = deque(maxlen=10)
        self.step_count = 0
        self.max_steps = 15000
        
        self.last_combo, self.prev_combo = 0, 0
        self.last_score, self.last_accuracy = 0, 1.0
        
        self.last_activity_time = time.time()
        self.user_quit = False
        self.activity_score = 0.0
        self.result_template = None
        self.game_ended_frames = 0
        
        print(f"✅ Environment initialized for osu!mania {self.num_keys}K mode.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f: return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def load_result_template(self, template_path: str):
        try:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.result_template = cv2.resize(template, (FRAME_SIZE, FRAME_SIZE))
                print(f"✅ Result screen template loaded from {template_path}")
        except Exception as e:
            print(f"❌ Error loading result template: {e}")

    def _get_state(self):
        try:
            sct_img = self.sct.grab(self.play_area)
            img = np.array(sct_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
        except Exception:
            return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

    def _read_ocr_value(self, area, value_type="int"):
        try:
            sct_img = self.sct.grab(area)
            img = np.array(sct_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            results = self.ocr_reader.readtext(gray, detail=0)
            for text in results:
                if value_type == "int":
                    numbers = re.findall(r'\d+', text.replace(',', ''))
                    if numbers: return int(numbers[0])
                elif value_type == "float":
                    match = re.search(r'(\d+\.?\d*)%?', text)
                    if match:
                        acc_val = float(match.group(1))
                        return acc_val / 100.0 if acc_val > 1 else acc_val
        except Exception: return None
        return None

    def _calculate_reward(self):
        reward = 0.0
        num_keys_pressed = sum(self.previous_keys_state)
        is_gameplay_active = self.activity_score > 0.01

        if is_gameplay_active:
            # 1. MISS
            if self.last_combo == 0 and self.prev_combo > 10:
                reward -= 5.0

            # 2. HIT
            score_diff = self.last_score - self.prev_score
            accuracy_diff = self.last_accuracy - self.prev_accuracy

            if score_diff > 0:
                if accuracy_diff >= -0.0001:
                    hit_reward = (0.5 + self.last_combo * 0.01) * self.last_accuracy
                    reward += hit_reward
                else:
                    reward += 0.1 
                    reward -= abs(accuracy_diff) * 2.0

            # 3. Punish if spam keys
            if num_keys_pressed > 0:
                reward -= 0.01 * num_keys_pressed
            else:
                reward -= 0.05

        else: # Menu
            if num_keys_pressed > 0:
                reward -= 0.5
            else:
                reward += 0.1

        self.prev_score = self.last_score
        self.prev_accuracy = self.last_accuracy

        return reward

    
    def _detect_game_activity(self, current_frame):
        if len(self.frame_buffer) < 2: return 0.0
        diff = cv2.absdiff(current_frame, self.frame_buffer[-1])
        activity_score = np.sum(diff > 25) / (FRAME_SIZE * FRAME_SIZE)
        if activity_score > 0.005: self.last_activity_time = time.time()
        return activity_score

    def _is_game_ended(self, current_frame) -> bool:
        if self.result_template is not None:
            try:
                res = cv2.matchTemplate(current_frame, self.result_template, cv2.TM_CCOEFF_NORMED)
                if np.max(res) > 0.8:
                    self.game_ended_frames += 1
                    if self.game_ended_frames > 5: return True
                else: self.game_ended_frames = 0
            except cv2.error: pass
        if self.user_quit: return True
        if time.time() - self.last_activity_time > 20.0 and self.last_combo == 0: return True
        return False

    def _execute_action_safely(self, action_combo):
        for i in range(self.num_keys):
            if self.previous_keys_state[i] != action_combo[i]:
                if action_combo[i]: pydirectinput.keyDown(self.keys[i])
                else: pydirectinput.keyUp(self.keys[i])
        self.previous_keys_state = action_combo.copy()

    def step(self, action):
        self.step_count += 1
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        self.prev_combo = self.last_combo
        if self.step_count % 4 == 0:
            self.last_combo = self._read_ocr_value(self.combo_area, "int") or self.last_combo
            self.last_score = self._read_ocr_value(self.score_area, "int") or self.last_score
            self.last_accuracy = self._read_ocr_value(self.accuracy_area, "float") or self.last_accuracy
        
        self._execute_action_safely(action_combo)
        time.sleep(FRAME_DELAY)
        
        new_frame = self._get_state()
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0)
        self.last_four_frames[-1] = new_frame
        
        self.activity_score = self._detect_game_activity(new_frame)
        self.frame_buffer.append(new_frame)
        
        reward = self._calculate_reward()
        
        terminated = self._is_game_ended(new_frame)
        truncated = self.step_count >= self.max_steps
        
        if self.show_window:
            self._show_enhanced_visualization(new_frame, action_combo, reward)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.user_quit = True

        info = {'current_combo': self.last_combo, 'ocr_accuracy': self.last_accuracy}
        return self.last_four_frames.copy(), reward, terminated, truncated, info

    def _show_enhanced_visualization(self, frame, action_combo, reward):
        vis_frame = cv2.cvtColor(cv2.resize(frame, (VISUALIZATION_SIZE, VISUALIZATION_SIZE)), cv2.COLOR_GRAY2BGR)
        key_width = VISUALIZATION_SIZE // self.num_keys
        for i in range(self.num_keys):
            x, color = i * key_width, ((0, 255, 0) if action_combo[i] else (0, 0, 255))
            cv2.rectangle(vis_frame, (x, 0), (x + key_width, 25), color, -1)
        
        y, h = 50, 25
        cv2.putText(vis_frame, f"Reward: {reward:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"Combo: {self.last_combo}", (10, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"Score: {self.last_score}", (10, y + 2*h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"OCR Acc: {self.last_accuracy*100:.1f}%", (10, y + 3*h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Mode: {self.num_keys}K", (VISUALIZATION_SIZE - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_frame, "Press 'q' in this window to quit", (10, VISUALIZATION_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow(f'Osu! Mania AI Agent - {self.num_keys}K', vis_frame)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("Resetting environment... Please start a new song.")

        for key in self.keys: 
            try: pydirectinput.keyUp(key)
            except: pass

        self.user_quit = False
        self.step_count, self.last_combo, self.prev_combo, self.last_score = 0, 0, 0, 0
        self.last_accuracy = 1.0
        self.last_activity_time = time.time()
        self.frame_buffer.clear()
        self.game_ended_frames = 0

        time.sleep(3)
        for i in range(4):
            frame = self._get_state()
            self.last_four_frames[i] = frame
            self.frame_buffer.append(frame)
            time.sleep(0.05)
        return self.last_four_frames.copy(), {}

    def close(self):
        cv2.destroyAllWindows()

        for key in self.keys: 
            try: pydirectinput.keyUp(key)
            except: pass
        print("Environment closed.")