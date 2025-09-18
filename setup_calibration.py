"""
Setup and calibration tool for Osu! Mania RL Agent
Helps determine correct play area coordinates and test capture
"""

import cv2
import mss
import numpy as np
import time
import json

class OsuSetupTool:
    def __init__(self):
        self.sct = mss.mss()
        self.play_area = None
        
    def show_full_screen(self):
        """Display full screen capture to help identify play area"""
        print("=== Full Screen Capture ===")
        print("Press 'q' to quit, 's' to save current coordinates")
        
        while True:
            # Capture full screen
            full_screen = self.sct.grab(self.sct.monitors[1])  # Primary monitor
            img = np.array(full_screen)
            
            # Convert to display format
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Scale down for display if too large
            height, width = img_rgb.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
            
            # Add coordinate info
            cv2.putText(img_rgb, f"Screen Size: {width}x{height}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_rgb, "Use this to identify osu! play area coordinates", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_rgb, "Press 'q' to continue", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow('Full Screen - Find Osu Play Area', img_rgb)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def interactive_area_selection(self):
        """Interactive tool to select play area"""
        print("\n=== Interactive Area Selection ===")
        print("1. Open osu! mania")
        print("2. Start a song or go to song select")
        print("3. Use the sliders below to adjust the capture area")
        
        input("Press Enter when osu! is ready...")
        
        # Default values - user will adjust these
        top = 230
        left = 710
        width = 490
        height = 750
        
        cv2.namedWindow('Area Selection', cv2.WINDOW_NORMAL)
        
        # Create trackbars
        cv2.createTrackbar('Top', 'Area Selection', top, 1080, lambda x: None)
        cv2.createTrackbar('Left', 'Area Selection', left, 1920, lambda x: None)
        cv2.createTrackbar('Width', 'Area Selection', width, 800, lambda x: None)
        cv2.createTrackbar('Height', 'Area Selection', height, 1000, lambda x: None)
        
        print("Adjust the trackbars until the red rectangle covers the play area")
        print("Press 's' to save, 'q' to quit")
        
        while True:
            # Get current values
            top = cv2.getTrackbarPos('Top', 'Area Selection')
            left = cv2.getTrackbarPos('Left', 'Area Selection')
            width = max(100, cv2.getTrackbarPos('Width', 'Area Selection'))
            height = max(100, cv2.getTrackbarPos('Height', 'Area Selection'))
            
            # Capture current area
            play_area = {'top': top, 'left': left, 'width': width, 'height': height}
            
            try:
                sct_img = self.sct.grab(play_area)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                
                # Scale up for better visibility
                display_img = cv2.resize(img, (width*2, height*2))
                
                # Add info overlay
                cv2.putText(display_img, f"Area: {left},{top} {width}x{height}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_img, "Adjust trackbars to fit play area", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display_img, "Press 's' to save, 'q' to quit", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Area Selection', display_img)
                
            except Exception as e:
                # Create error image if capture fails
                error_img = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(error_img, "Capture Error - Adjust coordinates", (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Area Selection', error_img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                self.play_area = play_area
                self.save_config()
                print(f"Saved configuration: {play_area}")
                break
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return self.play_area
    
    def test_capture_and_processing(self, play_area=None):
        """Test the capture and image processing"""
        if play_area is None:
            play_area = self.play_area or {'top': 230, 'left': 710, 'width': 490, 'height': 750}
        
        print("\n=== Testing Capture and Processing ===")
        print("This will show what the AI sees")
        print("Press 'q' to quit")
        
        cv2.namedWindow('AI Vision', cv2.WINDOW_NORMAL)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Capture like the AI does
                sct_img = self.sct.grab(play_area)
                img = np.array(sct_img)
                
                # Process like the AI does
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                resized = cv2.resize(gray, (84, 84))
                
                # Create visualization
                display_original = cv2.resize(img, (336, 336))  # 4x larger
                display_processed = cv2.resize(resized, (336, 336))
                display_processed = cv2.cvtColor(display_processed, cv2.COLOR_GRAY2BGR)
                
                # Combine both views
                combined = np.hstack([display_original[:,:,:3], display_processed])
                
                # Add labels and info
                cv2.putText(combined, "Original Capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(combined, "AI Processed (84x84)", (350, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add FPS counter
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                else:
                    fps = 0
                
                cv2.putText(combined, f"FPS: {fps:.1f}", (10, 370),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add activity detection visualization
                if hasattr(self, 'last_frame'):
                    diff = cv2.absdiff(resized, self.last_frame)
                    activity_score = np.sum(diff > 30) / (84 * 84)
                    cv2.putText(combined, f"Activity: {activity_score:.3f}", (350, 370),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                self.last_frame = resized.copy()
                
                cv2.imshow('AI Vision', combined)
                
            except Exception as e:
                print(f"Capture error: {e}")
                error_img = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error: {str(e)}", (50, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('AI Vision', error_img)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def save_config(self):
        """Save configuration to file"""
        if self.play_area:
            config = {
                'play_area': self.play_area,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('osu_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"Configuration saved to osu_config.json")
            
            # Also generate code snippet
            print("\nAdd this to your training script:")
            print(f"PLAY_AREA = {self.play_area}")
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open('osu_config.json', 'r') as f:
                config = json.load(f)
            self.play_area = config.get('play_area')
            print(f"Loaded configuration: {self.play_area}")
            return self.play_area
        except FileNotFoundError:
            print("No configuration file found")
            return None
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("=== Osu! Mania RL Agent Setup ===")
        print("This tool will help you configure the capture area")
        
        # Step 1: Load existing config if available
        existing_config = self.load_config()
        if existing_config:
            print("Found existing configuration")
            use_existing = input("Use existing configuration? (y/n): ").lower() == 'y'
            if use_existing:
                self.test_capture_and_processing(existing_config)
                return existing_config
        
        # Step 2: Show full screen for reference
        print("\nStep 1: Full screen view")
        self.show_full_screen()
        
        # Step 3: Interactive area selection
        print("\nStep 2: Select play area")
        play_area = self.interactive_area_selection()
        
        if play_area:
            # Step 4: Test the configuration
            print("\nStep 3: Testing configuration")
            self.test_capture_and_processing(play_area)
            
            print("\nSetup completed!")
            print("You can now run the training script with the saved configuration")
        
        return play_area

def main():
    setup_tool = OsuSetupTool()
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test":
            # Test with current or default config
            config = setup_tool.load_config()
            if not config:
                config = {'top': 230, 'left': 710, 'width': 490, 'height': 750}
            setup_tool.test_capture_and_processing(config)
        elif command == "select":
            # Just do area selection
            setup_tool.interactive_area_selection()
        else:
            print("Usage: python setup_calibration.py [test|select]")
    else:
        # Full setup
        setup_tool.run_full_setup()

if __name__ == '__main__':
    main()