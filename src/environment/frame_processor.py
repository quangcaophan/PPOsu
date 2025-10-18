"""
Frame processing utilities for the osu!mania environment.
Handles screen capture, preprocessing, and frame management.
"""

import cv2
import mss
import numpy as np
import time
import threading
from queue import Queue, Empty
from collections import deque
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .constants import FRAME_SIZE, TARGET_FPS, FRAME_DELAY, MAX_FRAME_QUEUE_SIZE


class FrameProcessor:
    """Handles frame capture and processing for the environment."""
    
    def __init__(self, play_area: Dict[str, int], target_fps: int = TARGET_FPS):
        """
        Initialize frame processor.
        
        Args:
            play_area: Screen area to capture (top, left, width, height)
            target_fps: Target frames per second
        """
        self.play_area = play_area
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Screen capture
        self.sct = mss.mss()
        
        # Frame processing
        self.crop_only = (
            play_area.get('width', FRAME_SIZE) == FRAME_SIZE and
            play_area.get('height', FRAME_SIZE) == FRAME_SIZE
        )
        
        # Threading
        self.frame_queue = Queue(maxsize=MAX_FRAME_QUEUE_SIZE)
        self.frame_intervals = deque(maxlen=120)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.capture_count = 0
        self.error_count = 0
        self.last_capture_time = 0
    
    def start(self) -> None:
        """Start the frame capture thread."""
        if self.running:
            return
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self) -> None:
        """Stop the frame capture thread."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest processed frame.
        
        Returns:
            Normalized frame array or None if no frame available
        """
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frame processing statistics."""
        avg_interval = np.mean(self.frame_intervals) if self.frame_intervals else 0
        current_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        
        return {
            "capture_count": self.capture_count,
            "error_count": self.error_count,
            "current_fps": current_fps,
            "queue_size": self.frame_queue.qsize(),
            "last_capture_time": self.last_capture_time
        }
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        next_frame_time = time.time()
        prev_time = time.time()
        
        while self.running:
            try:
                # Capture screen
                sct_img = self.sct.grab(self.play_area)
                img = np.array(sct_img)
                
                # Process frame
                processed_frame = self._process_frame(img)
                
                # Update queue
                if not self.frame_queue.full():
                    self.frame_queue.put(processed_frame)
                else:
                    # Remove old frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(processed_frame)
                    except Empty:
                        pass
                
                # Update statistics
                self.capture_count += 1
                interval = (time.time() - prev_time) * 1000
                self.frame_intervals.append(interval)
                prev_time = time.time()
                self.last_capture_time = time.time()
                
                # Maintain target FPS
                next_frame_time += self.frame_interval
                delay = next_frame_time - time.time()
                if delay > 0:
                    time.sleep(delay)
                
            except Exception as e:
                self.error_count += 1
                print(f"Frame capture error: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, img: np.ndarray) -> np.ndarray:
        """
        Process captured frame.
        
        Args:
            img: Raw captured image
            
        Returns:
            Processed and normalized frame
        """
        # Convert BGRA to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = img
        
        # Resize if needed
        if not self.crop_only:
            gray = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))
        
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        return normalized
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame synchronously.
        
        Returns:
            Processed frame or None if capture failed
        """
        try:
            sct_img = self.sct.grab(self.play_area)
            img = np.array(sct_img)
            return self._process_frame(img)
        except Exception as e:
            print(f"Single frame capture error: {e}")
            return None
    
    def test_performance(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Test frame capture performance.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Performance statistics
        """
        print(f"Testing frame capture performance for {duration} seconds...")
        
        start_time = time.time()
        frame_count = 0
        times = []
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            frame = self.capture_single_frame()
            frame_end = time.time()
            
            if frame is not None:
                frame_count += 1
                times.append(frame_end - frame_start)
        
        elapsed = time.time() - start_time
        avg_time = np.mean(times) if times else 0
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        stats = {
            "duration": elapsed,
            "frame_count": frame_count,
            "avg_capture_time": avg_time,
            "fps": fps,
            "target_fps": self.target_fps
        }
        
        print(f"Performance test results:")
        print(f"  FPS: {fps:.1f} (target: {self.target_fps})")
        print(f"  Avg capture time: {avg_time*1000:.2f}ms")
        print(f"  Frames captured: {frame_count}")
        
        return stats
