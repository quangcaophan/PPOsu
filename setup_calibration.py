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

FRAME_SIZE = 96

class OsuSetupTool:
    def __init__(self):
        self.sct = mss.mss()
        self.config = {
            'play_area': None,
            'combo_area': None,
            'score_area': None,
            'accuracy_area': None
        }
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
                
                left = int(min(x1, x2) / self.scale_factor)
                top = int(min(y1, y2) / self.scale_factor)
                width = int(abs(x2 - x1) / self.scale_factor)
                height = int(abs(y2 - y1) / self.scale_factor)

                area = {'top': top, 'left': left, 'width': width, 'height': height}
                print(f"✅ {name} selected: {area}")
                cv2.destroyWindow(window_name)
                return area
            elif key == ord('r'):
                self.start_point, self.end_point = (-1, -1), (-1, -1)
            elif key == ord('q'):
                print("Selection cancelled.")
                cv2.destroyWindow(window_name)
                return None

    def capture_result_template(self):
        if not self.config['play_area']:
            print("❌ Play area must be set first!")
            return
            
        print("\n--- Result Screen Template Capture ---")
        print("1. In osu!, finish a song to get to the result screen.")
        print("2. Once there, come back here and press Enter.")
        input("Press Enter when the result screen is visible...")
        
        try:
            template_img = np.array(self.sct.grab(self.config['play_area']))
            gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGRA2GRAY)

            os.makedirs("template", exist_ok=True)
            path = "template/result_template.png"
            cv2.imwrite(path, gray_template)
            print(f"✅ Result template saved to {path}")
            
            cv2.imshow("Template Preview", gray_template)
            print("Showing preview. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"❌ Error capturing template: {e}")

    def test_ocr_areas(self):
        if not all(self.config.values()):
            print("❌ All areas must be set before testing OCR.")
            return
            
        print("\n--- Testing OCR Areas ---")
        print("Go play a song in osu! to see live OCR results.")
        print("Press 'q' in the test window to stop.")
        
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=True)
        except ImportError:
            print("❌ EasyOCR not found. Please run: pip install easyocr")
            return

        while True:
            def get_ocr_image(area):
                img = np.array(self.sct.grab(area))
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                _, thresh = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
                return img, thresh

            combo_img, combo_proc = get_ocr_image(self.config['combo_area'])
            score_img, score_proc = get_ocr_image(self.config['score_area'])
            acc_img, acc_proc = get_ocr_image(self.config['accuracy_area'])

            combo_text = reader.readtext(combo_proc, detail=0)
            score_text = reader.readtext(score_proc, detail=0)
            acc_text = reader.readtext(acc_proc, detail=0)
            
            # Display logic
            h, w = 150, 250
            display = np.zeros((h, w*3, 3), dtype=np.uint8)
            display[0:100, 0:w] = cv2.resize(cv2.cvtColor(combo_img, cv2.COLOR_BGRA2BGR), (w, 100))
            display[0:100, w:w*2] = cv2.resize(cv2.cvtColor(score_img, cv2.COLOR_BGRA2BGR), (w, 100))
            display[0:100, w*2:w*3] = cv2.resize(cv2.cvtColor(acc_img, cv2.COLOR_BGRA2BGR), (w, 100))
            
            cv2.putText(display, f"Read: {combo_text}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(display, f"Read: {score_text}", (w+10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(display, f"Read: {acc_text}", (w*2+10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            cv2.imshow("OCR Test", display)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def test_capture_performance(self):
        if not self.config['play_area']:
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
            
            avg_time_ms = np.mean(times) * 1000
            fps = 1 / np.mean(times)
            
            status, color = ("EXCELLENT", (0,255,0)) if fps >= 55 else \
                            (("GOOD", (0,255,255)) if fps >= 45 else \
                            (("ACCEPTABLE", (0,165,255)) if fps >= 30 else \
                            ("POOR", (0,0,255))))

            display_img = cv2.cvtColor(cv2.resize(resized, (420, 420)), cv2.COLOR_GRAY2BGR)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display_img, f"Time: {avg_time_ms:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(display_img, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Performance Test", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        
    def save_configuration(self, filename="osu_config.json"):
        if not all(self.config.values()):
            print("❌ Not all areas are set. Cannot save.")
            return
        
        self.config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"\n✅ Configuration successfully saved to {filename}")

    def load_configuration(self, filename="osu_config.json"):
        if not os.path.exists(filename):
            print(f"Info: No configuration file found at {filename}.")
            return False
        with open(filename, 'r') as f:
            self.config = json.load(f)
        print(f"✅ Configuration loaded from {filename}")
        return True

    def run(self):
        print("="*60)
        print("🎮 ENHANCED OSU! MANIA RL AGENT SETUP TOOL 🎮")
        print("="*60)
        
        if self.load_configuration():
            if input("Load existing configuration? (y/n): ").lower() == 'y':
                if input("Test this configuration? (y/n): ").lower() == 'y':
                    self.test_capture_performance()
                    self.test_ocr_areas()
                return

        print("\nStarting new setup...")
        
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

        if input("\nCapture result screen template? (y/n): ").lower() == 'y':
            self.capture_result_template()
            
        print("\n--- Final Testing ---")
        self.test_capture_performance()
        self.test_ocr_areas()
        
        self.save_configuration()
        print("\n🎉 SETUP COMPLETED! You can now run the training script.")

if __name__ == '__main__':
    tool = OsuSetupTool()
    tool.run()