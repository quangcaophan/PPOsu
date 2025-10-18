"""
Setup tool for PPO Osu! Mania agent.
Interactive tool for configuring play area and other settings.
"""

import os
import sys
import cv2
import mss
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config_manager import ConfigManager, AgentConfig
from src.core.logger import setup_colored_logger


class OsuSetupTool:
    """Interactive setup tool for configuring the osu!mania agent."""
    
    def __init__(self):
        self.sct = mss.mss()
        self.config_manager = ConfigManager()
        self.logger = setup_colored_logger("setup_tool")
        
        # Setup state
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.scale_factor = 1.0
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for area selection."""
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
        """Capture and scale screen for display."""
        monitor = self.sct.monitors[2]  # Primary monitor
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
        """Interactive area selection with mouse."""
        window_name = f"{name} Selection"
        self.logger.info(f"\n--- {name} Setup ---")
        print("1. A screenshot will appear.")
        print("2. Click and drag to select the area.")
        print("3. Press 's' to save, 'r' to reset, 'q' to cancel.")
        input("Press Enter when you are ready...")
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        
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
                self.logger.info(f"Selected: {area}")
                cv2.destroyWindow(window_name)
                return area
            elif key == ord('r'):
                self.start_point = (-1, -1)
                self.end_point = (-1, -1)
            elif key == ord('q'):
                self.logger.info("Selection cancelled.")
                cv2.destroyWindow(window_name)
                return None
    
    def test_capture_performance(self, play_area: Dict[str, int]):
        """Test screen capture performance."""
        self.logger.info("\n--- Testing Capture Performance ---")
        print("Press 'q' in the test window to stop.")
        
        times = []
        while True:
            start_time = time.perf_counter()
            sct_img = self.sct.grab(play_area)
            img = np.array(sct_img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            resized = cv2.resize(gray, (128, 128))  # New frame size
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            if not times:
                continue
            
            fps = 1 / np.mean(times[-60:])  # Last 60 frames
            display_img = cv2.cvtColor(cv2.resize(resized, (420, 420)), cv2.COLOR_GRAY2BGR)
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Performance Test", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        self.logger.info("Performance test completed.")
    
    def create_config(self, mode: str, num_keys: int, play_area: Dict[str, int]) -> AgentConfig:
        """Create a new configuration."""
        config = self.config_manager.create_default_config(mode, num_keys)
        
        # Set play area
        config.play_area.top = play_area['top']
        config.play_area.left = play_area['left']
        config.play_area.width = play_area['width']
        config.play_area.height = play_area['height']
        
        # Set timestamp
        from datetime import datetime
        config.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return config
    
    def run(self):
        """Run the setup tool."""
        print("="*60)
        print("🎮 OSU! MANIA AI AGENT SETUP TOOL 🎮")
        print("="*60)
        
        # Select game mode
        print("\nSelect the game mode:")
        print("1. osu!mania")
        mode_choice = input("Enter your choice (1): ").strip()
        mode_name = "mania" if mode_choice == '1' else None
        
        if not mode_name:
            self.logger.error("Invalid selection. Aborting.")
            return
        
        # Select key mode
        key_mode_input = input("Enter the key mode for osu!mania (e.g., 4, 7): ").strip()
        try:
            key_mode = int(key_mode_input)
        except ValueError:
            self.logger.error("Invalid number. Aborting.")
            return
        
        if key_mode not in [4, 5, 6, 7]:
            self.logger.error("Unsupported key mode. Please choose 4, 5, 6, or 7.")
            return
        
        # Check if config already exists
        config_name = f"{mode_name}_{key_mode}k"
        if config_name in self.config_manager.list_configs():
            self.logger.info(f"\nConfiguration '{config_name}' already exists.")
            action = input("Choose an action: [T]est existing, [R]e-run Setup, [Q]uit: ").lower().strip()
            
            if action == 't':
                # Load existing config and test
                config = self.config_manager.load_config(config_name)
                self.test_capture_performance(config.play_area.__dict__)
                return
            elif action == 'q':
                return
        
        # Create new configuration
        self.logger.info(f"\nStarting new setup for osu!{mode_name} {key_mode}K...")
        
        # Select play area
        play_area = self._interactive_area_selection("Play Area")
        if not play_area:
            self.logger.error("Setup cancelled. Aborting.")
            return
        
        # Create configuration
        config = self.create_config(mode_name, key_mode, play_area)
        
        # Test performance
        self.logger.info("\n--- Final Testing ---")
        self.test_capture_performance(play_area)
        
        # Save configuration
        self.config_manager.save_config(config, config_name)
        
        self.logger.info(f"\n🎉 SETUP COMPLETED!")
        self.logger.info(f"Configuration saved as: {config_name}")
        self.logger.info(f"To train, run: python main.py --config {config_name}")
        self.logger.info(f"To play, run: python play_agent.py --config {config_name}")


def main():
    """Main entry point."""
    try:
        tool = OsuSetupTool()
        tool.run()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user.")
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
