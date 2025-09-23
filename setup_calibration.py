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

# Hằng số được đồng bộ với môi trường game
FRAME_SIZE = 96

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
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                area = {
                    'top': int(min(y1, y2) / self.scale_factor),
                    'left': int(min(x1, x2) / self.scale_factor),
                    'width': int(abs(x2 - x1) / self.scale_factor),
                    'height': int(abs(y2 - y1) / self.scale_factor)
                }
                print(f"✅ {name} selected: {area}")
                cv2.destroyWindow(window_name)
                return area
            elif key == ord('r'):
                self.start_point, self.end_point = (-1, -1), (-1, -1)
            elif key == ord('q'):
                print("Selection cancelled.")
                cv2.destroyWindow(window_name)
                return None

    def capture_result_template(self, mode_name: str, key_mode: Optional[int] = None):
        if not self.config.get('play_area'):
            print("❌ Play area must be set first!")
            return

        print("\n--- Result Screen Template Capture ---")
        print("1. In osu!, finish a song to get to the result screen.")
        input("Press Enter when the result screen is visible...")
        
        try:
            template_img = np.array(self.sct.grab(self.config['play_area']))
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGRA2GRAY)

            template_dir = f"templates/{mode_name}" + (f"_{key_mode}k" if key_mode else "")
            os.makedirs(template_dir, exist_ok=True)
            path = f"{template_dir}/result_template.png"
            
            cv2.imwrite(path, gray_template)
            print(f"✅ Result template saved to {path}")
            
            cv2.imshow("Template Preview", gray_template)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"❌ Error capturing template: {e}")

    def save_configuration(self, mode_name: str, key_mode: Optional[int] = None):
        if not self.config.get('play_area'):
            print("❌ Cannot save, play area is not set.")
            return
        
        os.makedirs("config", exist_ok=True)
        filename = f"config/{mode_name}" + (f"_{key_mode}k" if key_mode else "") + "_config.json"
        
        # Thêm thông tin về mode vào file config
        self.config['mode'] = mode_name
        if key_mode:
            self.config['num_keys'] = key_mode
        
        self.config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"\n✅ Configuration successfully saved to {filename}")

    def run(self):
        print("="*60)
        print("🎮 OSU! AI AGENT SETUP & CALIBRATION TOOL 🎮")
        print("="*60)
        
        # <<< HỎI NGƯỜI DÙNG MUỐN SETUP MODE NÀO
        print("Select the game mode you want to configure:")
        print("1. osu!mania")
        print("2. osu!taiko (Coming Soon)")
        mode_choice = input("Enter your choice (1): ").strip()
        
        mode_name = ""
        key_mode = None

        if mode_choice == '1':
            mode_name = "mania"
            while True:
                try:
                    key_mode_input = input("Enter the key mode for osu!mania (e.g., 4, 5, 6, 7): ").strip()
                    key_mode = int(key_mode_input)
                    if key_mode > 0:
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            print("Invalid selection. Aborting.")
            return

        print(f"\nStarting setup for osu!{mode_name} {key_mode}K...")
        
        setup_steps = [
            ("Play Area", "play_area"),
            ("Combo Area", "combo_area"),
            ("Score Area", "score_area"),
            ("Accuracy Area", "accuracy_area"),
        ]
        
        for name, key in setup_steps:
            area = self._interactive_area_selection(name)
            if area:
                self.config[key] = area
            else:
                print(f"❌ Setup cancelled during {name} selection. Aborting.")
                return

        if input("\nCapture result screen template for this mode? (y/n): ").lower() == 'y':
            self.capture_result_template(mode_name, key_mode)
            
        # Các bài test giữ nguyên vì chúng dùng chung logic
        print("\n--- Final Testing ---")
        # self.test_capture_performance()
        # self.test_ocr_areas()
        
        self.save_configuration(mode_name, key_mode)
        print("\n🎉 SETUP COMPLETED! You can now run the training script.")
        print(f"To train, run: python train.py --config config/{mode_name}_{key_mode}k_config.json")

if __name__ == '__main__':
    tool = OsuSetupTool()
    tool.run()