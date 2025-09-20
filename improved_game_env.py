import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
from collections import deque
import threading
import easyocr
import json
import os
from typing import Optional, Dict, Any, Tuple

# Định nghĩa các phím cho chế độ 4K
KEYS = ['d', 'f', 'j', 'k']

class ImprovedOsuManiaEnv(gym.Env):
    def __init__(self, play_area=None, num_keys=4, show_window=True, config_file="osu_config.json"):
        super(ImprovedOsuManiaEnv, self).__init__()

        # Load configuration
        self.config = self._load_config(config_file)
        self.play_area = play_area or self.config.get('play_area', {'top': 230, 'left': 710, 'width': 490, 'height': 750})
        
        self.sct = mss.mss()
        self.num_keys = num_keys
        self.show_window = show_window
        
        # Initialize OCR reader (supports Vietnamese and English)
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Define areas for different UI elements
        self._setup_ui_areas()
        
        # Action space: 2^num_keys possible combinations
        self.action_space = spaces.Discrete(2**self.num_keys)

        # Observation space: 4 consecutive grayscale frames
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, 84, 84), dtype=np.uint8)
        
        # State tracking
        self.last_four_frames = np.zeros((4, 84, 84), dtype=np.uint8)
        self.previous_keys_state = [False] * self.num_keys
        self.current_keys_state = [False] * self.num_keys
        
        # Performance tracking
        self.frame_buffer = deque(maxlen=10)
        self.step_count = 0
        self.max_steps = 15000  # Increased for longer songs
        
        # Enhanced reward tracking with OCR
        self.last_combo = 0
        self.last_score = 0
        self.combo_history = deque(maxlen=5)
        self.miss_count = 0
        self.hit_count = 0
        
        # Long note detection
        self.long_note_states = [False] * self.num_keys  # Which keys are in long notes
        self.long_note_start_frames = [0] * self.num_keys
        
        # Game state detection
        self.result_template = None
        self.game_ended_frames = 0
        self.last_activity_time = time.time()
        
        print(f"Environment initialized with play area: {self.play_area}")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            return {}
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_file}, using defaults")
            return {}

    def _setup_ui_areas(self):
        """Setup areas for UI elements like combo, score, etc."""
        # These coordinates are relative to the play area
        # You may need to adjust these based on your osu! skin
        base_top = self.play_area['top']
        base_left = self.play_area['left']
        base_width = self.play_area['width']
        
        # Combo area (usually top-left of play area)
        self.combo_area = {
            'top': base_top - 100,
            'left': base_left - 150,
            'width': 200,
            'height': 80
        }
        
        # Score area (usually top-right)
        self.score_area = {
            'top': base_top - 100,
            'left': base_left + base_width - 100,
            'width': 200,
            'height': 80
        }

    def _get_state(self):
        """Capture and preprocess game screen"""
        try:
            sct_img = self.sct.grab(self.play_area)
            img = np.array(sct_img)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            resized = cv2.resize(gray, (84, 84))
            
            return resized
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return np.zeros((84, 84), dtype=np.uint8)

    def _read_combo_ocr(self) -> int:
        """Read combo using OCR"""
        try:
            sct_img = self.sct.grab(self.combo_area)
            img = np.array(sct_img)
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
            # Enhance contrast for numbers
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold to get white text on black background
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Read text
            results = self.ocr_reader.readtext(thresh)
            
            # Look for combo number (usually format "123x" or just "123")
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Only accept confident readings
                    # Extract numbers from text
                    import re
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        combo = int(numbers[0])
                        if combo >= 0:  # Sanity check
                            return combo
            
            return self.last_combo  # Return last known combo if OCR fails
            
        except Exception as e:
            if self.step_count % 100 == 0:  # Don't spam error messages
                print(f"OCR Error (step {self.step_count}): {e}")
            return self.last_combo

    def _read_score_ocr(self) -> int:
        """Read score using OCR"""
        try:
            sct_img = self.sct.grab(self.score_area)
            img = np.array(sct_img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            results = self.ocr_reader.readtext(thresh)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    import re
                    # Remove commas and extract numbers
                    text_cleaned = text.replace(',', '')
                    numbers = re.findall(r'\d+', text_cleaned)
                    if numbers:
                        score = int(numbers[0])
                        if score >= self.last_score:  # Score should only increase
                            return score
            
            return self.last_score
            
        except Exception as e:
            return self.last_score

    def _detect_long_notes(self, current_frame) -> Dict[int, bool]:
        """Detect long notes in each column"""
        long_notes = {}
        width_per_key = 84 // self.num_keys
        
        for i in range(self.num_keys):
            left = i * width_per_key
            right = (i + 1) * width_per_key
            
            # Extract the entire column
            column = current_frame[:, left:right]
            
            # Look for vertical bright streaks (long notes)
            # Calculate vertical connectivity
            bright_pixels = column > 150
            vertical_streaks = []
            
            for col_idx in range(column.shape[1]):
                col_data = bright_pixels[:, col_idx]
                # Find continuous bright segments
                segments = []
                start = None
                for row_idx, is_bright in enumerate(col_data):
                    if is_bright and start is None:
                        start = row_idx
                    elif not is_bright and start is not None:
                        segments.append(row_idx - start)
                        start = None
                if start is not None:
                    segments.append(len(col_data) - start)
                
                # If we have a segment longer than 1/3 of the screen height, it's likely a long note
                max_segment = max(segments) if segments else 0
                if max_segment > 84 * 0.33:
                    vertical_streaks.append(max_segment)
            
            # If multiple columns in this key have long streaks, it's a long note
            long_notes[i] = len(vertical_streaks) > width_per_key * 0.5
            
        return long_notes

    def _calculate_reward_with_ocr(self, current_frame, action) -> float:
        """Enhanced reward calculation using OCR data"""
        reward = 0.0
        
        # 1. Read game state using OCR
        current_combo = self._read_combo_ocr()
        current_score = self._read_score_ocr()
        
        # Update combo history
        self.combo_history.append(current_combo)
        
        # 2. Primary reward: Combo-based feedback
        if current_combo > self.last_combo:
            # Successful hit! This is the most important signal
            hit_reward = 5.0 + (current_combo * 0.1)  # Bonus for longer combos
            reward += hit_reward
            self.hit_count += 1
            
            # Extra bonus for hitting during detected activity
            activity_score = self._detect_game_activity(current_frame)
            if activity_score > 0.02:
                reward += 2.0  # Reward for hitting when there's actually something to hit
                
        elif current_combo == 0 and self.last_combo > 0:
            # Combo break (miss) - this is a strong negative signal
            miss_penalty = -15.0 - (self.last_combo * 0.2)  # Harder penalty for breaking long combos
            reward += miss_penalty
            self.miss_count += 1
            
        elif current_combo == self.last_combo and current_combo > 0:
            # Combo maintained, small positive reinforcement
            reward += 0.5
        
        # 3. Score-based reward (secondary)
        score_diff = current_score - self.last_score
        if score_diff > 0:
            reward += score_diff / 10000.0  # Small reward for score increase
        
        # 4. Long note handling
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        detected_long_notes = self._detect_long_notes(current_frame)
        
        for i in range(self.num_keys):
            # Long note start detection
            if detected_long_notes[i] and not self.long_note_states[i]:
                # Starting a long note
                if action_combo[i]:
                    reward += 3.0  # Good long note start
                    self.long_note_states[i] = True
                    self.long_note_start_frames[i] = self.step_count
                else:
                    reward -= 2.0  # Missed long note start
            
            # Long note continuation
            elif self.long_note_states[i]:
                if detected_long_notes[i]:
                    # Long note is still active
                    if action_combo[i]:
                        reward += 0.2  # Good continuation
                    else:
                        reward -= 3.0  # Released too early
                        self.long_note_states[i] = False
                else:
                    # Long note ended
                    if not action_combo[i]:
                        # Good release timing
                        hold_duration = self.step_count - self.long_note_start_frames[i]
                        reward += 2.0 + (hold_duration * 0.01)
                    else:
                        reward -= 1.0  # Held too long
                    self.long_note_states[i] = False
        
        # 5. Activity-based penalties for wrong actions
        activity_score = self._detect_game_activity(current_frame)
        num_keys_pressed = sum(action_combo)
        
        # Penalty for pressing keys when no activity (but don't punish during long notes)
        if activity_score < 0.005 and num_keys_pressed > 0 and not any(self.long_note_states):
            reward -= 1.0 * num_keys_pressed
        
        # 6. Inactivity penalty
        if activity_score < 0.001 and sum(self.combo_history) == 0:
            reward -= 0.1  # Small penalty for complete inactivity
        
        # Update state for next step
        self.last_combo = current_combo
        self.last_score = current_score
        
        return reward

    def _detect_game_activity(self, current_frame):
        """Detect if there's activity in the game (notes moving)"""
        if len(self.frame_buffer) < 3:
            return 0
        
        # Compare current frame with previous frames
        diff_scores = []
        for prev_frame in list(self.frame_buffer)[-3:]:
            diff = cv2.absdiff(current_frame, prev_frame)
            activity_score = np.sum(diff > 30) / (84 * 84)
            diff_scores.append(activity_score)
        
        activity = np.mean(diff_scores)
        
        # Update activity tracking
        if activity > 0.01:
            self.last_activity_time = time.time()
        
        return activity

    def _is_game_ended(self, current_frame) -> bool:
        """Enhanced game end detection using template matching"""
        # Method 1: Template matching (if result template is available)
        if self.result_template is not None:
            try:
                result = cv2.matchTemplate(current_frame, self.result_template, cv2.TM_CCOEFF_NORMED)
                max_val = np.max(result)
                
                if max_val > 0.8:  # High similarity with result screen
                    self.game_ended_frames += 1
                    return self.game_ended_frames > 5  # Require multiple consecutive frames
                else:
                    self.game_ended_frames = 0
                    
            except Exception as e:
                pass  # Fall back to other methods
        
        # Method 2: Extended inactivity
        time_since_activity = time.time() - self.last_activity_time
        combo_inactive = len(self.combo_history) > 0 and max(self.combo_history) == 0
        
        if time_since_activity > 10.0 and combo_inactive:  # 10 seconds of no activity
            return True
        
        # Method 3: OCR-based detection (look for "Retry" or "Back" buttons)
        try:
            # Check bottom area for result screen UI
            ui_area = {
                'top': self.play_area['top'] + self.play_area['height'] - 100,
                'left': self.play_area['left'],
                'width': self.play_area['width'],
                'height': 100
            }
            
            sct_img = self.sct.grab(ui_area)
            img = np.array(sct_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
            results = self.ocr_reader.readtext(gray)
            
            for (bbox, text, confidence) in results:
                if confidence > 0.7:
                    text_lower = text.lower()
                    if any(keyword in text_lower for keyword in ['retry', 'back', 'ranking', 'result']):
                        return True
                        
        except Exception as e:
            pass
        
        return False

    def step(self, action):
        self.step_count += 1
        
        # 1. Execute action with improved timing
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # Handle key state changes more smoothly
        for i in range(self.num_keys):
            if self.previous_keys_state[i] and not action_combo[i]:
                pydirectinput.keyUp(KEYS[i])
            elif not self.previous_keys_state[i] and action_combo[i]:
                pydirectinput.keyDown(KEYS[i])
        
        self.previous_keys_state = action_combo.copy()
        self.current_keys_state = action_combo.copy()
        
        # Frame timing (targeting 60 FPS)
        time.sleep(0.016)
        
        # 2. Get new state
        new_frame = self._get_state()
        self.frame_buffer.append(new_frame.copy())
        
        # Update frame stack
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0)
        self.last_four_frames[-1] = new_frame
        
        # 3. Calculate reward using enhanced system
        reward = self._calculate_reward_with_ocr(new_frame, action)
        
        # 4. Check termination conditions
        done = (self.step_count >= self.max_steps) or self._is_game_ended(new_frame)
        truncated = self.step_count >= self.max_steps
        
        # 5. Show visualization if enabled
        if self.show_window:
            self._show_enhanced_visualization(new_frame, action_combo, reward)
        
        # 6. Comprehensive info
        info = {
            'step_count': self.step_count,
            'current_combo': self.last_combo,
            'current_score': self.last_score,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'accuracy': self.hit_count / max(self.hit_count + self.miss_count, 1),
            'activity_score': self._detect_game_activity(new_frame),
            'keys_pressed': action_combo,
            'long_note_states': self.long_note_states.copy(),
            'reward_components': {
                'total': reward,
                'combo': self.last_combo,
                'score_diff': self.last_score - getattr(self, 'prev_score', 0)
            }
        }
        
        return self.last_four_frames.copy(), reward, done, truncated, info

    def _show_enhanced_visualization(self, frame, action_combo, reward):
        """Enhanced visualization with OCR data and long note info"""
        # Create larger visualization
        vis_frame = cv2.resize(frame, (420, 420))  # 5x larger
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw key indicators
        key_width = 420 // self.num_keys
        for i in range(self.num_keys):
            x = i * key_width
            
            # Color coding: Green=pressed, Red=released, Blue=long note
            if self.long_note_states[i]:
                color = (255, 0, 0)  # Blue for long notes
            elif action_combo[i]:
                color = (0, 255, 0)  # Green for pressed
            else:
                color = (0, 0, 255)  # Red for released
                
            cv2.rectangle(vis_frame, (x, 0), (x + key_width - 1, 25), color, -1)
            cv2.putText(vis_frame, KEYS[i].upper(), (x + 10, 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Enhanced info display
        info_y = 50
        line_height = 25
        
        cv2.putText(vis_frame, f"Reward: {reward:.2f}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(vis_frame, f"Combo: {self.last_combo}", (10, info_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(vis_frame, f"Score: {self.last_score}", (10, info_y + 2*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        accuracy = self.hit_count / max(self.hit_count + self.miss_count, 1) * 100
        cv2.putText(vis_frame, f"Accuracy: {accuracy:.1f}%", (10, info_y + 3*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        cv2.putText(vis_frame, f"Hits: {self.hit_count} | Miss: {self.miss_count}", (10, info_y + 4*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"Step: {self.step_count}", (10, info_y + 5*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw hit zones and long note indicators
        hit_zone_top = int(420 * 0.6)
        hit_zone_bottom = int(420 * 0.8)
        cv2.rectangle(vis_frame, (0, hit_zone_top), (420, hit_zone_bottom), (255, 0, 0), 2)
        
        # Long note indicators
        for i, is_long_note in enumerate(self.long_note_states):
            if is_long_note:
                x = i * key_width
                cv2.putText(vis_frame, "LN", (x + 5, hit_zone_top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('Enhanced Osu! Mania AI Agent', vis_frame)
        cv2.waitKey(1)

    def load_result_template(self, template_path: str):
        """Load result screen template for better game end detection"""
        try:
            self.result_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if self.result_template is not None:
                self.result_template = cv2.resize(self.result_template, (84, 84))
                print(f"Result template loaded from {template_path}")
            else:
                print(f"Could not load result template from {template_path}")
        except Exception as e:
            print(f"Error loading result template: {e}")

    def reset(self, seed=None, options=None):
        """Reset the environment with enhanced initialization"""
        super().reset(seed=seed)
        
        print("Resetting environment...")
        print("Please start a new song in osu! mania...")
        
        # Release all keys
        for key in KEYS:
            pydirectinput.keyUp(key)
        
        # Reset all state variables
        self.previous_keys_state = [False] * self.num_keys
        self.current_keys_state = [False] * self.num_keys
        self.step_count = 0
        self.last_combo = 0
        self.last_score = 0
        self.hit_count = 0
        self.miss_count = 0
        self.combo_history.clear()
        self.frame_buffer.clear()
        self.long_note_states = [False] * self.num_keys
        self.long_note_start_frames = [0] * self.num_keys
        self.game_ended_frames = 0
        self.last_activity_time = time.time()
        
        # Wait for game to start
        time.sleep(3)
        
        # Capture initial frames
        for i in range(4):
            new_frame = self._get_state()
            self.last_four_frames[i] = new_frame
            self.frame_buffer.append(new_frame.copy())
            time.sleep(0.05)
        
        # Initial OCR reading to establish baseline
        self.last_combo = self._read_combo_ocr()
        self.last_score = self._read_score_ocr()
        
        info = {
            'step_count': 0,
            'initial_combo': self.last_combo,
            'initial_score': self.last_score
        }
        return self.last_four_frames.copy(), info

    def close(self):
        """Clean up resources"""
        # Release all keys
        for key in KEYS:
            pydirectinput.keyUp(key)
        
        # Close visualization window
        cv2.destroyAllWindows()
        
        print("Enhanced environment closed successfully.")
        print(f"Final stats - Hits: {self.hit_count}, Misses: {self.miss_count}")
        if self.hit_count + self.miss_count > 0:
            accuracy = self.hit_count / (self.hit_count + self.miss_count) * 100
            print(f"Final accuracy: {accuracy:.1f}%")