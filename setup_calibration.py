"""
Automated Setup and Calibration Tool for Osu! Mania RL Agent
Enhanced version with mouse selection and template capture
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import mss
import numpy as np
import time
import json
from typing import Dict, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class OsuSetupTool:
    def __init__(self):
        self.sct = mss.mss()
        self.play_area = None
        self.combo_area = None
        self.score_area = None
        self.accuracy_area = None
        
        # Mouse selection state
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.current_image = None
        self.scale_factor = 1.0
        
        # Templates for game state detection
        self.result_template = None
        
    def mouse_callback(self, event, x, y):
        """Handle mouse events for area selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
    
    def capture_full_screen(self) -> np.ndarray:
        """Capture and scale full screen for display"""
        full_screen = self.sct.grab(self.sct.monitors[1])  # Primary monitor
        img = np.array(full_screen)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # Scale down if too large
        height, width = img_rgb.shape[:2]
        max_display_size = 1200
        
        if width > max_display_size or height > max_display_size:
            self.scale_factor = min(max_display_size/width, max_display_size/height)
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        else:
            self.scale_factor = 1.0
        
        return img_rgb
    
    def interactive_area_selection(self, window_name: str, instruction: str) -> Optional[Dict[str, int]]:
        """Generic interactive area selection with mouse"""
        print(f"\n=== {instruction} ===")
        print("1. The full screen will be displayed")
        print("2. Click and drag to select the area")
        print("3. Press 's' to save the selection")
        print("4. Press 'r' to reset selection") 
        print("5. Press 'q' to skip/cancel")
        
        input("Press Enter when ready...")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.drawing = False
        
        selected_area = None
        
        while True:
            # Get fresh screenshot
            screen_img = self.capture_full_screen()
            display_img = screen_img.copy()
            
            # Draw selection rectangle
            if self.start_point != (-1, -1) and self.end_point != (-1, -1):
                cv2.rectangle(display_img, self.start_point, self.end_point, (0, 255, 0), 2)
                
                # Show coordinates
                coord_text = f"({self.start_point[0]}, {self.start_point[1]}) to ({self.end_point[0]}, {self.end_point[1]})"
                cv2.putText(display_img, coord_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(display_img, instruction, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_img, "Click and drag to select area", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_img, "'s' to save, 'r' to reset, 'q' to cancel", (10, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s') and self.start_point != (-1, -1) and self.end_point != (-1, -1):
                # Convert screen coordinates to actual coordinates
                actual_start_x = int(self.start_point[0] / self.scale_factor)
                actual_start_y = int(self.start_point[1] / self.scale_factor)
                actual_end_x = int(self.end_point[0] / self.scale_factor)
                actual_end_y = int(self.end_point[1] / self.scale_factor)
                
                # Ensure top-left to bottom-right
                left = min(actual_start_x, actual_end_x)
                top = min(actual_start_y, actual_end_y)
                right = max(actual_start_x, actual_end_x)
                bottom = max(actual_start_y, actual_end_y)
                
                selected_area = {
                    'top': top,
                    'left': left,
                    'width': right - left,
                    'height': bottom - top
                }
                print(f"Selected area: {selected_area}")
                break
                
            elif key == ord('r'):
                # Reset selection
                self.start_point = (-1, -1)
                self.end_point = (-1, -1)
                self.drawing = False
                
            elif key == ord('q'):
                print("Selection cancelled")
                break
        
        cv2.destroyWindow(window_name)
        return selected_area
    
    def setup_play_area(self) -> Optional[Dict[str, int]]:
        """Setup main play area"""
        print("=== Play Area Setup ===")
        print("Please open osu! mania and start a song or go to song select")
        print("You need to select the rectangular area where the notes fall")
        
        self.play_area = self.interactive_area_selection(
            "Play Area Selection", 
            "Select the main play area (where notes fall)"
        )
        return self.play_area
    
    def setup_combo_area(self) -> Optional[Dict[str, int]]:
        """Setup combo reading area"""
        if not self.play_area:
            print("Play area must be set first!")
            return None
            
        print("=== Combo Area Setup ===")
        print("Select the area where the combo number is displayed")
        print("This is usually in the top-left area of the screen")
        
        self.combo_area = self.interactive_area_selection(
            "Combo Area Selection",
            "Select where the combo number is displayed"
        )
        return self.combo_area
    
    def setup_score_area(self) -> Optional[Dict[str, int]]:
        """Setup score reading area"""
        print("=== Score Area Setup ===")
        print("Select the area where the score is displayed")
        print("This is usually in the top-right area of the screen")
        
        self.score_area = self.interactive_area_selection(
            "Score Area Selection",
            "Select where the score is displayed"
        )
        return self.score_area
    
    def setup_accuracy_area(self) -> Optional[Dict[str, int]]:
        """Setup accuracy reading area"""
        print("=== Accuracy Setup ===")
        print("Select the area where the accuracy is displayed")
        print("This is usually in the top-right area of the screen")
        
        self.accuracy_area = self.interactive_area_selection(
            "Accuracy Area Selection",
            "Select where the accuracy is displayed"
        )
        return self.accuracy_area
    
    def capture_result_template(self) -> bool:
        """Capture result screen template for game end detection"""
        print("=== Result Screen Template Capture ===")
        print("1. Play a short song in osu! mania")
        print("2. When you reach the result screen, come back to this window")
        print("3. The tool will capture the result screen as a template")
        
        input("Press Enter when you're at the result screen...")
        
        try:
            if self.play_area:
                # Capture the play area as template
                sct_img = self.sct.grab(self.play_area)
                img = np.array(sct_img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                
                # Save the template
                template_path = r"template\result_template.png"
                cv2.imwrite(template_path, gray)
                
                # Also save a preview for user to verify
                preview_path = r"template\result_template_preview.png"
                cv2.imwrite(preview_path, img)
                
                print(f"Result template saved as {template_path}")
                print(f"Preview saved as {preview_path}")
                
                # Show preview
                cv2.imshow("Result Template Preview", gray)
                print("Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                return True
            else:
                print("Play area not set! Cannot capture template.")
                return False
                
        except Exception as e:
            print(f"Error capturing result template: {e}")
            return False
    
    def test_ocr_areas(self) -> bool:
        """Test OCR on combo and score areas"""
        if not self.combo_area or not self.score_area or not self.accuracy_area:
            print("Combo, score and accuracy areas must be set first!")
            return False
        
        print("=== Testing OCR Areas ==...")
        print("Press 'q' to stop testing")
        
        try:
            import re
            import easyocr
            reader = easyocr.Reader(['en'])
        except ImportError:
            print("easyocr not installed! Install with: pip install easyocr")
            return False
        
        cv2.namedWindow("OCR Test", cv2.WINDOW_NORMAL)
        
        while True:
            try:
                combo_img_raw = self.sct.grab(self.combo_area)
                score_img_raw = self.sct.grab(self.score_area)
                accuracy_img_raw = self.sct.grab(self.accuracy_area)
                
                combo_img = np.array(combo_img_raw)
                score_img = np.array(score_img_raw)
                accuracy_img = np.array(accuracy_img_raw)
                
                combo_gray = cv2.cvtColor(combo_img, cv2.COLOR_BGRA2GRAY)
                score_gray = cv2.cvtColor(score_img, cv2.COLOR_BGRA2GRAY)
                accuracy_gray = cv2.cvtColor(accuracy_img, cv2.COLOR_BGRA2GRAY)
                
                combo_processed = cv2.threshold(combo_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                score_processed = cv2.threshold(score_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                
                accuracy_processed_for_ocr = accuracy_gray
                
                combo_results = reader.readtext(combo_processed)
                score_results = reader.readtext(score_processed)
                accuracy_results = reader.readtext(accuracy_processed_for_ocr)
                
                combo_text = "No detect"
                score_text = "No detect"
                accuracy_text = "No detect"
                
                if combo_results:
                    numbers = re.findall(r'\d+', combo_results[0][1])
                    if numbers: combo_text = f"Combo: {numbers[0]}"

                if score_results:
                    numbers = re.findall(r'\d+', score_results[0][1].replace(',', ''))
                    if numbers: score_text = f"Score: {numbers[0]}"
                
                if accuracy_results:
                    numbers = re.findall(r'[\d.]+', accuracy_results[0][1])
                    if numbers and numbers[0] != '.':
                        accuracy_text = f"Acc: {numbers[0]}"

                combo_display = cv2.resize(cv2.cvtColor(combo_img, cv2.COLOR_BGRA2BGR), (200, 100))
                score_display = cv2.resize(cv2.cvtColor(score_img, cv2.COLOR_BGRA2BGR), (200, 100))
                accuracy_display = cv2.resize(cv2.cvtColor(accuracy_img, cv2.COLOR_BGRA2BGR), (200, 100))
                
                display = np.zeros((200, 600, 3), dtype=np.uint8)
                display[0:100, 0:200] = combo_display
                display[0:100, 200:400] = score_display
                display[0:100, 400:600] = accuracy_display
                
                cv2.putText(display, combo_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, score_text, (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, accuracy_text, (410, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, "Press 'q' to stop", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                cv2.imshow("OCR Test", display)
                
            except Exception as e:
                print(f"Loop error: {e}")
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return True
    
    def test_capture_performance(self) -> bool:
        """Test capture performance and frame rate"""
        if not self.play_area:
            print("Play area must be set first!")
            return False
        
        print("=== Testing Capture Performance ===")
        print("Testing frame rate and processing speed...")
        print("Press 'q' to stop")
        
        cv2.namedWindow("Performance Test", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        start_time = time.time()
        processing_times = []

        display_fps = 0
        
        while True:
            frame_start = time.time()
            
            try:
                # Simulate the full processing pipeline
                sct_img = self.sct.grab(self.play_area)
                img = np.array(sct_img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                resized = cv2.resize(gray, (84, 84))
                
                frame_end = time.time()
                processing_time = (frame_end - frame_start) * 1000  # ms
                processing_times.append(processing_time)
                
                # Keep only recent measurements
                if len(processing_times) > 60:
                    processing_times.pop(0)
                
                frame_count += 1
                elapsed = time.time() - start_time
                
                if elapsed > 0.5:
                    display_fps = frame_count / elapsed
                    avg_processing = np.mean(processing_times)
                    
                    # Performance indicator
                    if display_fps >= 55:
                        status = "EXCELLENT"
                        color = (0, 255, 0)
                    elif display_fps >= 45:
                        status = "GOOD"
                        color = (0, 255, 255)
                    elif display_fps >= 30:
                        status = "ACCEPTABLE"
                        color = (0, 165, 255)
                    else:
                        status = "POOR"
                        color = (0, 0, 255)
                    
                    # Reset counters
                    frame_count = 0
                    start_time = time.time()

                avg_processing = np.mean(processing_times) if processing_times else 0

                # Create visualization
                display_img = cv2.resize(resized, (420, 420))
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                
                # Add performance info
                cv2.putText(display_img, f"FPS: {display_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, f"Processing: {avg_processing:.1f}ms", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_img, f"Target: 60 FPS (16.7ms)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(display_img, f"Performance: {status}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(display_img, "Press 'q' to stop", (10, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Performance Test", display_img)
                
            except Exception as e:
                error_img = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error: {str(e)}", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Performance Test", error_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if processing_times:
            avg_time = np.mean(processing_times)
            print(f"Average processing time: {avg_time:.2f}ms")
            print(f"Estimated max FPS: {1000/avg_time:.1f}")
            
            if avg_time < 16.7:
                print("✅ Performance is good for 60 FPS training")
            else:
                print("⚠️ Performance may be too slow for optimal training")
        
        return True
    
    def save_configuration(self, filename: str = "osu_config.json") -> bool:
        """Save all configuration to JSON file"""
        if not all([self.play_area, self.combo_area, self.score_area, self.accuracy_area]):
            print("All areas must be configured before saving!")
            return False
        
        config = {
            'play_area': self.play_area,
            'combo_area': self.combo_area,
            'score_area': self.score_area,
            'accuracy_area': self.accuracy_area,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '2.0'
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"✅ Configuration saved to {filename}")
            
            # Generate code snippets for easy copy-paste
            print("\n" + "="*50)
            print("COPY THESE VALUES TO YOUR TRAINING SCRIPT:")
            print("="*50)
            print(f"PLAY_AREA = {self.play_area}")
            print(f"COMBO_AREA = {self.combo_area}")
            print(f"SCORE_AREA = {self.score_area}")
            print(f"ACCURACY_AREA = {self.accuracy_area}")
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_configuration(self, filename: str = "osu_config.json") -> bool:
        """Load configuration from JSON file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            self.play_area = config.get('play_area')
            self.combo_area = config.get('combo_area')  
            self.score_area = config.get('score_area')
            self.accuracy_area = config.get('accuracy_area')
            
            print(f"✅ Configuration loaded from {filename}")
            print(f"Play area: {self.play_area}")
            print(f"Combo area: {self.combo_area}")
            print(f"Score area: {self.score_area}")
            print(f"Accuracy area: {self.accuracy_area}")
            
            return True
            
        except FileNotFoundError:
            print(f"Configuration file {filename} not found")
            return False
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        print("="*60)
        print("🎮 ENHANCED OSU! MANIA RL AGENT SETUP TOOL")
        print("="*60)
        
        # Check if config exists
        if os.path.exists("osu_config.json"):
            print("Found existing configuration file.")
            use_existing = input("Load existing configuration? (y/n): ").lower().strip()
            
            if use_existing == 'y':
                if self.load_configuration():
                    # Still offer to test
                    test_existing = input("Test existing configuration? (y/n): ").lower().strip()
                    if test_existing == 'y':
                        self.test_capture_performance()
                        self.test_ocr_areas()
                    return True
        
        print("\nStarting fresh setup...")
        print("Make sure osu! mania is open and ready!")
        
        # Step 1: Play area
        print("\n" + "="*40)
        print("STEP 1: PLAY AREA SETUP")
        print("="*40)
        if not self.setup_play_area():
            print("❌ Play area setup failed!")
            return False
        
        # Step 2: Combo area  
        print("\n" + "="*40)
        print("STEP 2: COMBO AREA SETUP")
        print("="*40)
        if not self.setup_combo_area():
            print("❌ Combo area setup failed!")
            return False
        
        # Step 3: Score area
        print("\n" + "="*40)
        print("STEP 3: SCORE AREA SETUP") 
        print("="*40)
        if not self.setup_score_area():
            print("❌ Score area setup failed!")
            return False
        
        # Step 4: Accuracy area
        print("\n" + "="*40)
        print("STEP 4: ACCURACY AREA SETUP") 
        print("="*40)
        if not self.setup_accuracy_area():
            print("❌ Accuracy area setup failed!")
            return False
        
        print("\n" + "="*40)
        print("RESULT TEMPLATE (OPTIONAL)")
        print("="*40)
        capture_template = input("Capture result screen template? (y/n): ").lower().strip()
        if capture_template == 'y':
            self.capture_result_template()
        
        # Testing
        print("\n" + "="*40)
        print("TESTING CONFIGURATION")
        print("="*40)
        
        print("Testing capture performance...")
        self.test_capture_performance()
        
        print("Testing OCR areas...")  
        self.test_ocr_areas()
        
        # Save configuration
        print("\n" + "="*40)
        print("SAVE CONFIGURATION")
        print("="*40)
        
        if self.save_configuration():
            print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("You can now run the training script with the saved configuration.")
            print("The training script will automatically load these settings.")
            return True
        else:
            print("❌ Failed to save configuration!")
            return False

def main():
    setup_tool = OsuSetupTool()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Test existing configuration
            if setup_tool.load_configuration():
                print("Testing performance...")
                setup_tool.test_capture_performance()
                print("Testing OCR...")
                setup_tool.test_ocr_areas()
            else:
                print("No configuration found to test!")
                
        elif command == "areas":
            # Just setup areas
            setup_tool.setup_play_area()
            setup_tool.setup_combo_area()
            setup_tool.setup_score_area()
            setup_tool.setup_accuracy_area()
            setup_tool.save_configuration()
            
        elif command == "template":
            # Just capture result template
            if setup_tool.load_configuration():
                setup_tool.capture_result_template()
            else:
                print("Load configuration first!")
                
        elif command == "performance":
            # Just test performance
            if setup_tool.load_configuration():
                setup_tool.test_capture_performance()
            else:
                print("Load configuration first!")
                
        else:
            print("Usage: python enhanced_setup_tool.py [test|areas|template|performance]")
    else:
        # Run complete setup
        setup_tool.run_complete_setup()

if __name__ == '__main__':
    main()