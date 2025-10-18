"""
Visualization utilities for the PPO Osu! Mania agent.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.logger import get_logger


class VisualizationManager:
    """Manages visualization windows and rendering."""
    
    def __init__(self, window_name: str = "Osu! Mania AI"):
        """
        Initialize visualization manager.
        
        Args:
            window_name: Name of the main window
        """
        self.window_name = window_name
        self.logger = get_logger("visualization")
        self.windows = {}
        
    def create_window(
        self, 
        name: str, 
        width: int = 420, 
        height: int = 420
    ) -> None:
        """
        Create a visualization window.
        
        Args:
            name: Window name
            width: Window width
            height: Window height
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)
        self.windows[name] = {"width": width, "height": height}
        self.logger.info(f"Created window: {name}")
    
    def show_frame(
        self, 
        window_name: str, 
        frame: np.ndarray, 
        overlay_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Display a frame in a window.
        
        Args:
            window_name: Name of the window
            frame: Frame to display
            overlay_info: Information to overlay on frame
        """
        if window_name not in self.windows:
            self.create_window(window_name)
        
        display_frame = frame.copy()
        
        # Add overlay information
        if overlay_info:
            display_frame = self._add_overlay(display_frame, overlay_info)
        
        cv2.imshow(window_name, display_frame)
    
    def _add_overlay(
        self, 
        frame: np.ndarray, 
        info: Dict[str, Any]
    ) -> np.ndarray:
        """Add information overlay to frame."""
        overlay_frame = frame.copy()
        
        # Convert to BGR if grayscale
        if len(overlay_frame.shape) == 2:
            overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_GRAY2BGR)
        
        y_offset = 30
        line_height = 25
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                overlay_frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            y_offset += line_height
        
        return overlay_frame
    
    def draw_key_indicators(
        self, 
        frame: np.ndarray, 
        action_combo: List[bool],
        key_width: int
    ) -> np.ndarray:
        """
        Draw key press indicators on frame.
        
        Args:
            frame: Frame to draw on
            action_combo: List of key states
            key_width: Width of each key indicator
            
        Returns:
            Frame with key indicators
        """
        display_frame = frame.copy()
        
        # Convert to BGR if grayscale
        if len(display_frame.shape) == 2:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        
        for i, pressed in enumerate(action_combo):
            x = i * key_width
            color = (0, 255, 0) if pressed else (0, 0, 255)
            cv2.rectangle(
                display_frame, (x, 0), (x + key_width, 25), color, -1
            )
        
        return display_frame
    
    def close_window(self, window_name: str) -> None:
        """Close a specific window."""
        if window_name in self.windows:
            cv2.destroyWindow(window_name)
            del self.windows[window_name]
            self.logger.info(f"Closed window: {window_name}")
    
    def close_all_windows(self) -> None:
        """Close all windows."""
        cv2.destroyAllWindows()
        self.windows.clear()
        self.logger.info("Closed all windows")
    
    def check_user_quit(self) -> bool:
        """Check if user wants to quit."""
        return (cv2.waitKey(1) & 0xFF) == ord('q')
    
    def wait_for_key(self, timeout: int = 0) -> int:
        """
        Wait for key press.
        
        Args:
            timeout: Timeout in milliseconds (0 = wait forever)
            
        Returns:
            Key code pressed
        """
        return cv2.waitKey(timeout) & 0xFF
