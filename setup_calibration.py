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
                x1, y1 = self.start_point; x2, y2 = self.end_point
                area = {
                    'top': int(min(y1, y2) / self.scale_factor), 'left': int(min(x1, x2) / self.scale_factor),
                    'width': int(abs(x2 - x1) / self.scale_factor), 'height': int(abs(y2 - y1) / self.scale_factor)
                }
                print(f"✅ {name} selected: {area}"); cv2.destroyWindow(window_name); return area
            elif key == ord('r'): self.start_point, self.end_point = (-1, -1), (-1, -1)
            elif key == ord('q'): print("Selection cancelled."); cv2.destroyWindow(window_name); return None

    def capture_result_template(self, mode_name: str, key_mode: Optional[int] = None):
        print("\n--- Result Screen Template Capture ---")
        print("1. In osu!, finish a song to get to the result screen.")
        print("2. IMPORTANT: Select a small, STATIC area on the result screen (e.g., a button, a corner).")
        
        template_area = self._interactive_area_selection("Result Screen Template")
        if not template_area: print("❌ Template capture cancelled."); return

        try:
            img = np.array(self.sct.grab(template_area))
            gray_template = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            template_dir = f"templates/{mode_name}" + (f"_{key_mode}k" if key_mode else "")
            os.makedirs(template_dir, exist_ok=True)
            path = f"{template_dir}/result_template.png"
            cv2.imwrite(path, gray_template)
            print(f"✅ Result template saved to {path}")
            cv2.imshow("Template Preview", gray_template); cv2.waitKey(0); cv2.destroyAllWindows()
        except Exception as e: print(f"❌ Error capturing template: {e}")

    # <<< KHÔI PHỤC LẠI HÀM test_capture_performance
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

    # <<< KHÔI PHỤC LẠI HÀM test_ocr_areas
    def test_ocr_areas(self):
        if not all(self.config.get(k) for k in ['play_area', 'combo_area', 'score_area', 'accuracy_area']):
            print("❌ All areas must be set before testing OCR.")
            return
        print("\n--- Testing OCR Areas ---")
        print("Go play a song in osu! to see live OCR results. Press 'q' to stop.")
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True)
        except ImportError:
            print("❌ EasyOCR not found. Please run: pip install easyocr"); return
        while True:
            try:
                combo_img = np.array(self.sct.grab(self.config['combo_area']))
                score_img = np.array(self.sct.grab(self.config['score_area']))
                acc_img = np.array(self.sct.grab(self.config['accuracy_area']))
                combo_text = reader.readtext(cv2.cvtColor(combo_img, cv2.COLOR_BGRA2GRAY), detail=0)
                score_text = reader.readtext(cv2.cvtColor(score_img, cv2.COLOR_BGRA2GRAY), detail=0)
                acc_text = reader.readtext(cv2.cvtColor(acc_img, cv2.COLOR_BGRA2GRAY), detail=0)
                h, w = 150, 250
                display = np.zeros((h, w*3, 3), dtype=np.uint8)
                display[0:100, 0:w] = cv2.resize(cv2.cvtColor(combo_img, cv2.COLOR_BGRA2BGR), (w, 100))
                display[0:100, w:w*2] = cv2.resize(cv2.cvtColor(score_img, cv2.COLOR_BGRA2BGR), (w, 100))
                display[0:100, w*2:w*3] = cv2.resize(cv2.cvtColor(acc_img, cv2.COLOR_BGRA2BGR), (w, 100))
                cv2.putText(display, f"Read: {combo_text}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(display, f"Read: {score_text}", (w+10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(display, f"Read: {acc_text}", (w*2+10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.imshow("OCR Test", display)
                if cv2.waitKey(100) & 0xFF == ord('q'): break
            except Exception: continue
        cv2.destroyAllWindows()

    def save_configuration(self, filename: str):
        if not self.config.get('play_area'): print("❌ Cannot save, play area is not set."); return
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
                self.test_ocr_areas()
                print("✅ Testing complete."); return
            elif action == 'q': return

        print(f"\nStarting new setup for osu!{mode_name} {key_mode}K...")
        self.config = {'mode': mode_name, 'num_keys': key_mode}
        
        setup_steps = [("Play Area", "play_area"), ("Combo Area", "combo_area"), 
                       ("Score Area", "score_area"), ("Accuracy Area", "accuracy_area")]
        
        for name, key in setup_steps:
            area = self._interactive_area_selection(name)
            if area: self.config[key] = area
            else: print(f"❌ Setup cancelled. Aborting."); return

        if input("\nCapture result screen template? (y/n): ").lower() == 'y':
            self.capture_result_template(mode_name, key_mode)
            
        print("\n--- Final Testing ---")
        self.test_capture_performance()
        self.test_ocr_areas()
        
        self.save_configuration(config_filename)
        print("\n🎉 SETUP COMPLETED!")
        print(f"To train, run: python train.py --config {config_filename}")

if __name__ == '__main__':
    tool = OsuSetupTool()
    tool.run()