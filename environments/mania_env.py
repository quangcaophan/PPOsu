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
from queue import Queue
from environments.constants import KEY_MAPPINGS, FRAME_SIZE, VISUALIZATION_SIZE, FRAME_DELAY

from memory_reader import UltraFastMemoryReader as MemoryReader

class OsuManiaEnv(gym.Env):
    """
    Optimized osu!mania Environment và Performance Optimizations
    """

    def __init__(self, config_path: str, show_window=False, run_id: str = "default"):
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
        self.key_hold_steps = np.zeros(self.num_keys, dtype=int)

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
        
        self.frame_queue = Queue(maxsize=1)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.latest_frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
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

    def _capture_loop(self):
        """Background thread - create separate mss instance"""
        import mss
        
        try:
            sct_thread = mss.mss(with_cursor=False)  # Thread-local mss
            print("✅ Capture thread started")
            
            while True:
                try:
                    frame_start = time.time()
                    
                    sct_img = sct_thread.grab(self.play_area)  # Use thread-local
                    img = np.array(sct_img)
                    resized = cv2.resize(img, (FRAME_SIZE, FRAME_SIZE))
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
                    
                    self.latest_frame = gray
                    capture_time = (time.time() - frame_start) * 1000
                    self.frame_times.append(capture_time)
                    
                    time.sleep(0.016)  # ~60 FPS
                    
                except Exception as e:
                    print(f"❌ Capture error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Capture thread init failed: {e}")

    def _get_state(self):
        """Get state - now just returns pre-captured frame (instant!)"""
        # This is now instant - frame was already captured in background
        return self.latest_frame.copy()

    def _calculate_reward(self):
        """
        Improved reward function with better signal for learning
        """
        reward = 0.0
        num_keys_pressed = sum(self.previous_keys_state)
        is_gameplay_active = self.game_state == 2

        if is_gameplay_active:
            if self.prev_hits:
                # Calculate hit differences
                hit_diffs = {k: self.last_hits.get(k, 0) - self.prev_hits.get(k, 0) for k in self.last_hits}
                
                # Calculate Rewards
                reward += hit_diffs.get('hit_geki', 0) * 5.0     # Perfect hits (increased)
                reward += hit_diffs.get('hit_300', 0) * 3.0      # Good hits (increased)
                reward += hit_diffs.get('hit_100', 0) * 1.0      # OK hits (less penalty)
                reward += hit_diffs.get('hit_50', 0) * -0.5      # Bad hits (small penalty)
                reward += hit_diffs.get('miss', 0) * -3.0        # Misses (reduced from -5.0)
                
                # Combo reward (encourage streaks)
                if self.last_combo > self.prev_combo:
                    combo_increase = self.last_combo - self.prev_combo
                    reward += combo_increase * 0.1  # Reward for extending combo
                
                # Small penalty for missing combo, but not crushing
                if self.last_combo == 0 and self.prev_combo > 5:
                    reward += -1.0  # Reduced from -5.0
            
            # CRITICAL FIX: Reward attempting to play, penalize staying idle
            if num_keys_pressed > 0:
                reward += 0.01  # Small positive for trying (instead of penalty)
            else:
                reward -= 0.02  # Small penalty for inaction
        
        else:
            pass # No penalty in menu, just neutral

        # Update previous values
        self.prev_combo = self.last_combo
        self.prev_score = self.last_score
        self.prev_accuracy = self.last_accuracy
        self.prev_hits = self.last_hits.copy()
        
        return reward

    def _is_game_ended(self) -> bool:
        """
        Check if game has ended based on game state from memory reader.
        State 2 is the gameplay screen.
        """
        return self.game_state != 2 or self.user_quit

    def _execute_action_safely(self, action_combo):
        """
        Execute key actions with THREADING to avoid blocking
        Non-blocking key presses using background thread
        """
        # Check if state changed
        for i in range(self.num_keys):
            if self.previous_keys_state[i] != action_combo[i]:
                # Use threading to avoid blocking
                key = self.keys[i]
                is_press = action_combo[i]
                
                # Create a thread for this key action
                import threading
                
                def send_key():
                    try:
                        if is_press:
                            pydirectinput.keyDown(key)
                        else:
                            pydirectinput.keyUp(key)
                    except Exception:
                        pass
                
                thread = threading.Thread(target=send_key, daemon=True)
                thread.start()
                # DON'T wait for thread - let it run in background
        
        self.previous_keys_state = action_combo.copy()

    def step(self, action):
        """Main step function with DETAILED TIMING"""
        step_start = time.time()
        self.step_count += 1
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]
        
        # TIMING EACH SECTION
        times = {}
        
        # 1. Execute action
        t1 = time.time()
        self._execute_action_safely(action_combo)
        times['action_execution'] = (time.time() - t1) * 1000
        
        # 2. Capture frame
        t2 = time.time()
        new_frame = self._get_state()
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0)
        self.last_four_frames[-1] = new_frame
        times['frame_capture'] = (time.time() - t2) * 1000
        
        # 3. Get game state from RAM
        t3 = time.time()
        game_state = self.memory_reader.get_game_state()
        self.game_state = game_state.get('game_state')
        self.last_score = game_state.get('score')
        self.last_combo = game_state.get('combo')
        self.last_accuracy = game_state.get('accuracy')
        self.last_hits = {k: v for k, v in game_state.items() if 'hit' in k or k == 'miss'}
        times['memory_read'] = (time.time() - t3) * 1000
        
        # 4. Calculate reward
        t4 = time.time()
        reward = self._calculate_reward()
        times['reward_calc'] = (time.time() - t4) * 1000
        
        # 5. Check termination
        t5 = time.time()
        terminated = self._is_game_ended()
        truncated = self.step_count >= self.max_steps
        times['termination_check'] = (time.time() - t5) * 1000
        
        # 6. Maintain FPS
        t6 = time.time()
        elapsed = time.time() - step_start
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)
        times['fps_maintain'] = (time.time() - t6) * 1000
        
        # 7. Visualization
        t7 = time.time()
        if self.show_window:
            self._show_visualization(new_frame, action_combo, reward, step_start)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.user_quit = True
        times['visualization'] = (time.time() - t7) * 1000
        
        total_time = (time.time() - step_start) * 1000
        times['total'] = total_time
        
        # PRINT TIMING BREAKDOWN every 200 steps
        if self.step_count % 200 == 0:
            print(f"\n=== STEP {self.step_count} TIMING BREAKDOWN ===")
            print(f"Action Execution:  {times['action_execution']:6.2f}ms")
            print(f"Frame Capture:     {times['frame_capture']:6.2f}ms")
            print(f"Memory Read:       {times['memory_read']:6.2f}ms")
            print(f"Reward Calc:       {times['reward_calc']:6.2f}ms")
            print(f"Termination Check: {times['termination_check']:6.2f}ms")
            print(f"FPS Maintain:      {times['fps_maintain']:6.2f}ms")
            print(f"Visualization:     {times['visualization']:6.2f}ms")
            print(f"{'─'*40}")
            print(f"TOTAL:             {times['total']:6.2f}ms ({1000/times['total']:.1f} FPS)")
            print(f"Unaccounted:       {times['total'] - sum([times[k] for k in times if k != 'total']):6.2f}ms")

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
        VIS_SIZE = 280
        vis_frame = cv2.cvtColor(cv2.resize(frame, (VIS_SIZE, VIS_SIZE)), cv2.COLOR_GRAY2BGR)
        key_width = VIS_SIZE // self.num_keys
        
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
        self.key_hold_steps.fill(0)
        
        # Start new trace file for the episode
        if hasattr(self, 'tracer'):
            self.tracer.start_episode()

        time.sleep(3)

        max_wait = 15  # seconds
        waited = 0
        while self.game_state != 2 and waited < max_wait:
            game_state_data = self.memory_reader.get_game_state()
            self.game_state = game_state_data.get('game_state')
            time.sleep(0.5)
            waited += 0.5
        
        if self.game_state != 2:
            self.log(f"WARNING: Game state is {self.game_state}, not 2 (Playing)", "WARNING")
        
        
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

