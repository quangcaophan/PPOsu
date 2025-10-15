import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import mss
import numpy as np
import time
import json
from typing import Dict, Optional
from collections import deque
from environments.constants import FRAME_SIZE


class OsuSetupTool:
    def __init__(self):
        self.sct = mss.mss()
        self.config = {}
        self.drawing = False
        self.start_point, self.end_point = (-1, -1), (-1, -1)
        self.scale_factor = 1.0

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

    def _capture_and_scale_screen(self):
        monitor = self.sct.monitors[1]
        img = np.array(self.sct.grab(monitor))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = img_bgr.shape[:2]
        max_display = 1200
        if h > max_display or w > max_display:
            self.scale_factor = min(max_display/h, max_display/w)
            return cv2.resize(img_bgr, (int(w*self.scale_factor), int(h*self.scale_factor)))
        else:
            self.scale_factor = 1.0
            return img_bgr

    def _interactive_area_selection(self, name: str) -> Optional[Dict[str, int]]:
        window_name = f"{name} Selection"
        print(f"\n--- {name} Setup ---")
        print("1. A screenshot will appear.")
        print("2. Click and drag to select the area.")
        print("3. Press 's' to save, 'r' to reset, 'q' to cancel.")
        input("Press Enter when you are ready...")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        self.start_point, self.end_point = (-1, -1), (-1, -1)
        
        while True:
            display_img = self._capture_and_scale_screen()
            if self.start_point != (-1, -1):
                cv2.rectangle(display_img, self.start_point, self.end_point, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') and self.start_point != (-1, -1):
                x1, y1 = self.start_point; x2, y2 = self.end_point
                area = {
                    'top': int(min(y1, y2) / self.scale_factor), 'left': int(min(x1, x2) / self.scale_factor),
                    'width': int(abs(x2 - x1) / self.scale_factor), 'height': int(abs(y2 - y1) / self.scale_factor)
                }
                print(f"✅ {name} selected: {area}"); cv2.destroyWindow(window_name); return area
            elif key == ord('r'): self.start_point, self.end_point = (-1, -1), (-1, -1)
            elif key == ord('q'): print("Selection cancelled."); cv2.destroyWindow(window_name); return None

    def test_capture_performance(self):
        if not self.config.get('play_area'):
            print("❌ Play area must be set first!")
            return
        print("\n--- Testing Capture Performance ---")
        print("Press 'q' in the test window to stop.")
        times = deque(maxlen=60)
        while True:
            start_time = time.perf_counter()
            sct_img = self.sct.grab(self.config['play_area'])
            img = np.array(sct_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            if not times: continue
            fps = 1 / np.mean(times)
            display_img = cv2.cvtColor(cv2.resize(resized, (420, 420)), cv2.COLOR_GRAY2BGR)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Performance Test", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

    def save_configuration(self, filename: str):
        if not self.config.get('play_area'): print("❌ Cannot save, play area is not set."); return
        
        self.config['max_steps'] = 15000
        self.config['reward_params'] = {
            "living_penalty": -0.01,
            "action_cost_penalty": -0.005,
            "idle_penalty": -0.05,
            
            "hit_geki_reward": 5.0,
            "hit_300_reward": 3.0,
            "hit_100_reward": 1.0,
            "hit_50_penalty": -0.5,
            "miss_penalty": -1.0,
            
            "combo_break_penalty": -0.5,
            "combo_increase_reward": 0.1,
            
            "combo_milestone_50": 10.0,
            "combo_milestone_100": 20.0,
            "combo_milestone_200": 40.0,
            
            "accuracy_change_multiplier": 100.0
        }
        self.config['training_params'] = {
            "policy": "CnnPolicy",
            "total_timesteps": 100000,
            "ppo_params": {
                "n_steps": 1024,
                "batch_size": 64,
                "ent_coef": 0.01,
                "learning_rate": 0.0001,
                "clip_range": 0.2,
                "n_epochs": 4
            },
            "callback_params": {
                "checkpoint_save_freq": 10000,
                "eval_freq": 5000,
                "n_eval_episodes": 5
            }
        }
        
        
        self.config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, 'w') as f: json.dump(self.config, f, indent=4)
        print(f"\n✅ Configuration successfully saved to {filename}")

    def load_configuration(self, filename: str):
        try:
            with open(filename, 'r') as f: self.config = json.load(f)
            print(f"✅ Configuration loaded from {filename}"); return True
        except FileNotFoundError: return False

    def run(self):
        print("="*60 + "\n🎮 OSU! AI AGENT SETUP & CALIBRATION TOOL 🎮\n" + "="*60)
        print("Select the game mode you want to work with:\n1. osu!mania")
        mode_choice = input("Enter your choice (1): ").strip()
        mode_name = "mania" if mode_choice == '1' else None
        if not mode_name: print("Invalid selection. Aborting."); return
        
        key_mode_input = input("Enter the key mode for osu!mania (e.g., 4, 7): ").strip()
        try: key_mode = int(key_mode_input)
        except ValueError: print("Invalid number. Aborting."); return

        os.makedirs("config", exist_ok=True)
        config_filename = f"config/{mode_name}_{key_mode}k_config.json"

        if self.load_configuration(config_filename):
            print("\n--- Existing Configuration Found ---")
            action = input("Choose an action: [T]est, [R]e-run Setup, [Q]uit: ").lower().strip()
            if action == 't':
                print("\n--- Starting Tests ---")
                self.test_capture_performance()
                print("✅ Testing complete."); return
            elif action == 'q': return

        print(f"\nStarting new setup for osu!{mode_name} {key_mode}K...")
        self.config = {'mode': mode_name, 'num_keys': key_mode}
        
        # The only required area is the play area for the agent's vision
        play_area = self._interactive_area_selection("Play Area")
        if play_area:
            self.config['play_area'] = play_area
        else:
            print(f"❌ Setup cancelled. Aborting."); return
            
        print("\n--- Final Testing ---")
        self.test_capture_performance()
        
        self.save_configuration(config_filename)
        print("\n🎉 SETUP COMPLETED!")
        print(f"To train, run: python train.py --config {config_filename}")

if __name__ == '__main__':
    tool = OsuSetupTool()
    tool.run()
