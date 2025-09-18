import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
from collections import deque
import threading

# Định nghĩa các phím cho chế độ 4K
KEYS = ['d', 'f', 'j', 'k']

class OsuManiaEnv(gym.Env):
    def __init__(self, play_area, num_keys=4, show_window=True):
        super(OsuManiaEnv, self).__init__()

        self.play_area = play_area
        self.sct = mss.mss()
        self.num_keys = num_keys
        self.show_window = show_window
        
        # Action space: 2^num_keys possible combinations
        self.action_space = spaces.Discrete(2**self.num_keys)

        # Observation space: 4 consecutive grayscale frames
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, 84, 84), dtype=np.uint8)
        
        # State tracking
        self.last_four_frames = np.zeros((4, 84, 84), dtype=np.uint8)
        self.previous_keys_state = [False] * self.num_keys
        
        # Performance tracking
        self.frame_buffer = deque(maxlen=10)  # Store recent frames for comparison
        self.step_count = 0
        self.max_steps = 10000  # Maximum steps per episode
        
        # Reward tracking
        self.last_activity_score = 0
        self.inactivity_penalty = 0

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

    def _detect_game_activity(self, current_frame):
        """Detect if there's activity in the game (notes moving)"""
        if len(self.frame_buffer) < 3:
            return 0
        
        # Compare current frame with previous frames
        diff_scores = []
        for prev_frame in list(self.frame_buffer)[-3:]:
            diff = cv2.absdiff(current_frame, prev_frame)
            activity_score = np.sum(diff > 30) / (84 * 84)  # Percentage of changed pixels
            diff_scores.append(activity_score)
        
        return np.mean(diff_scores)

    def _calculate_reward(self, current_frame, action):
        """Calculate reward based on game state and action"""
        reward = 0
        
        # 1. Activity-based reward
        activity_score = self._detect_game_activity(current_frame)
        
        # Reward for game activity (notes moving)
        if activity_score > 0.01:  # Threshold for detecting movement
            reward += 1.0
            self.inactivity_penalty = max(0, self.inactivity_penalty - 0.1)
        else:
            # Penalty for inactivity (no notes detected)
            self.inactivity_penalty = min(2.0, self.inactivity_penalty + 0.05)
            reward -= self.inactivity_penalty
        
        # 2. Action consistency reward
        # Penalize excessive key pressing when no activity
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        num_keys_pressed = sum(action_combo)
        
        if activity_score < 0.005 and num_keys_pressed > 0:
            reward -= 0.5 * num_keys_pressed  # Penalty for pressing keys when no notes
        
        # 3. Encourage varied actions when there's activity
        if activity_score > 0.02 and num_keys_pressed > 0:
            reward += 0.2 * num_keys_pressed
        
        # 4. Detect potential hits by looking at bright spots in key areas
        hit_reward = self._detect_potential_hits(current_frame, action_combo)
        reward += hit_reward
        
        return reward

    def _detect_potential_hits(self, frame, action_combo):
        """Detect potential note hits based on bright spots in key columns"""
        reward = 0
        
        # Divide frame into key columns (assuming 4 keys)
        width_per_key = 84 // self.num_keys
        hit_zone_bottom = int(84 * 0.8)  # Bottom 20% of screen (typical hit zone)
        hit_zone_top = int(84 * 0.6)    # Top of hit zone
        
        for i in range(self.num_keys):
            # Define column boundaries
            left = i * width_per_key
            right = (i + 1) * width_per_key
            
            # Extract hit zone for this key
            key_zone = frame[hit_zone_top:hit_zone_bottom, left:right]
            
            # Check for bright spots (notes)
            bright_pixels = np.sum(key_zone > 150)  # Threshold for bright pixels
            total_pixels = key_zone.size
            brightness_ratio = bright_pixels / total_pixels
            
            # If there's a potential note and key is pressed
            if brightness_ratio > 0.1 and action_combo[i]:
                reward += 2.0  # Good hit reward
            elif brightness_ratio > 0.1 and not action_combo[i]:
                reward -= 1.0  # Missed note penalty
            elif brightness_ratio < 0.05 and action_combo[i]:
                reward -= 0.5  # Wrong key press penalty
        
        return reward

    def _is_game_ended(self, current_frame):
        """Check if the game/song has ended"""
        # This is a simple heuristic - you might need to adjust based on osu! UI
        # Look for specific patterns that indicate game over
        
        # Check if the screen is mostly static (no activity for extended period)
        if hasattr(self, 'static_frame_count'):
            activity = self._detect_game_activity(current_frame)
            if activity < 0.001:
                self.static_frame_count += 1
            else:
                self.static_frame_count = 0
            
            # If no activity for 100 frames, consider game ended
            return self.static_frame_count > 100
        else:
            self.static_frame_count = 0
            return False

    def step(self, action):
        self.step_count += 1
        
        # 1. Execute action with proper timing
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # Release keys that should not be pressed
        for i in range(self.num_keys):
            if self.previous_keys_state[i] and not action_combo[i]:
                pydirectinput.keyUp(KEYS[i])
        
        # Press keys that should be pressed
        for i in range(self.num_keys):
            if not self.previous_keys_state[i] and action_combo[i]:
                pydirectinput.keyDown(KEYS[i])
        
        self.previous_keys_state = action_combo.copy()
        
        # Small delay to match game timing (adjust based on your needs)
        time.sleep(0.016)  # ~60 FPS
        
        # 2. Get new state
        new_frame = self._get_state()
        self.frame_buffer.append(new_frame.copy())
        
        # Update frame stack
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0)
        self.last_four_frames[-1] = new_frame
        
        # 3. Calculate reward
        reward = self._calculate_reward(new_frame, action)
        
        # 4. Check termination conditions
        done = (self.step_count >= self.max_steps) or self._is_game_ended(new_frame)
        truncated = self.step_count >= self.max_steps
        
        # 5. Show visualization if enabled
        if self.show_window:
            self._show_visualization(new_frame, action_combo, reward)
        
        info = {
            'step_count': self.step_count,
            'activity_score': self._detect_game_activity(new_frame),
            'keys_pressed': action_combo,
            'reward_components': {
                'total': reward,
                'activity': self._detect_game_activity(new_frame),
                'action_count': sum(action_combo)
            }
        }
        
        return self.last_four_frames.copy(), reward, done, truncated, info

    def _show_visualization(self, frame, action_combo, reward):
        """Display visualization window"""
        # Create a larger visualization
        vis_frame = cv2.resize(frame, (336, 336))  # 4x larger
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw key indicators
        key_width = 336 // self.num_keys
        for i in range(self.num_keys):
            x = i * key_width
            color = (0, 255, 0) if action_combo[i] else (0, 0, 255)
            cv2.rectangle(vis_frame, (x, 0), (x + key_width - 1, 20), color, -1)
            cv2.putText(vis_frame, KEYS[i].upper(), (x + 10, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw reward info
        cv2.putText(vis_frame, f"Reward: {reward:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(vis_frame, f"Step: {self.step_count}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw hit zones
        hit_zone_top = int(336 * 0.6)
        hit_zone_bottom = int(336 * 0.8)
        cv2.rectangle(vis_frame, (0, hit_zone_top), (336, hit_zone_bottom), (255, 0, 0), 2)
        
        cv2.imshow('Osu! Mania AI Agent', vis_frame)
        cv2.waitKey(1)

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        print("Resetting environment...")
        print("Please start a new song in osu! mania...")
        
        # Release all keys
        for key in KEYS:
            pydirectinput.keyUp(key)
        
        self.previous_keys_state = [False] * self.num_keys
        self.step_count = 0
        self.inactivity_penalty = 0
        self.frame_buffer.clear()
        
        if hasattr(self, 'static_frame_count'):
            self.static_frame_count = 0
        
        # Wait for game to start
        time.sleep(3)
        
        # Capture initial frames
        for i in range(4):
            new_frame = self._get_state()
            self.last_four_frames[i] = new_frame
            self.frame_buffer.append(new_frame.copy())
            time.sleep(0.05)
        
        info = {'step_count': 0}
        return self.last_four_frames.copy(), info

    def close(self):
        """Clean up resources"""
        # Release all keys
        for key in KEYS:
            pydirectinput.keyUp(key)
        
        # Close visualization window
        cv2.destroyAllWindows()
        
        print("Environment closed successfully.")