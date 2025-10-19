"""
Auto beatmap selector for training with multiple songs.
Automatically selects a new random beatmap after each song.
"""

import random
import time
import os
from pathlib import Path
from typing import List, Optional
import pydirectinput
from stable_baselines3.common.callbacks import BaseCallback

from ..core.logger import get_logger


class AutoBeatmapSelector(BaseCallback):
    """
    Automatically selects random beatmaps during training.
    Changes to a new map after each song completes.
    """
    
    def __init__(
        self,
        beatmap_folder: str,
        min_difficulty: float = 1.0,
        max_difficulty: float = 3.0,
        key_mode: int = 4,
        verbose: int = 1
    ):
        """
        Initialize auto beatmap selector.
        
        Args:
            beatmap_folder: Path to osu! Songs folder (e.g. "C:/osu!/Songs")
            min_difficulty: Minimum star rating
            max_difficulty: Maximum star rating
            key_mode: Number of keys (4, 5, 6, 7)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.beatmap_folder = Path(beatmap_folder)
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.key_mode = key_mode
        self.logger = get_logger("beatmap_selector")
        
        # State tracking
        self.current_song_count = 0
        self.beatmap_history = []
        self.max_history = 10  # Don't repeat last 10 songs
        
        # Find available beatmaps
        self.available_beatmaps = self._scan_beatmaps()
        
        if not self.available_beatmaps:
            self.logger.warning(f"No {key_mode}K beatmaps found in {beatmap_folder}")
        else:
            self.logger.info(
                f"Found {len(self.available_beatmaps)} available {key_mode}K beatmaps "
                f"(difficulty {min_difficulty}-{max_difficulty}★)"
            )
    
    def _scan_beatmaps(self) -> List[str]:
        """Scan beatmap folder for suitable maps."""
        if not self.beatmap_folder.exists():
            self.logger.error(f"Beatmap folder not found: {self.beatmap_folder}")
            return []
        
        suitable_maps = []
        
        try:
            # Scan all song folders
            for song_folder in self.beatmap_folder.iterdir():
                if not song_folder.is_dir():
                    continue
                
                # Look for .osu files
                for osu_file in song_folder.glob("*.osu"):
                    map_info = self._parse_osu_file(osu_file)
                    
                    if map_info and self._is_suitable(map_info):
                        suitable_maps.append({
                            'path': osu_file,
                            'name': map_info.get('title', 'Unknown'),
                            'difficulty': map_info.get('difficulty_name', 'Unknown'),
                            'stars': map_info.get('stars', 0)
                        })
        
        except Exception as e:
            self.logger.error(f"Error scanning beatmaps: {e}")
        
        return suitable_maps
    
    def _parse_osu_file(self, osu_file: Path) -> Optional[dict]:
        """Parse .osu file to get map info."""
        try:
            with open(osu_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            info = {}
            
            # Extract metadata
            for line in content.split('\n'):
                line = line.strip()
                
                if line.startswith('Title:'):
                    info['title'] = line.split(':', 1)[1].strip()
                elif line.startswith('Version:'):
                    info['difficulty_name'] = line.split(':', 1)[1].strip()
                elif line.startswith('CircleSize:'):
                    # CircleSize in mania = number of keys
                    cs = float(line.split(':')[1].strip())
                    info['keys'] = int(cs)
                elif line.startswith('OverallDifficulty:'):
                    # Rough star estimation (not accurate but ok for filtering)
                    od = float(line.split(':')[1].strip())
                    info['stars'] = od * 0.6  # Rough approximation
            
            return info
        
        except Exception as e:
            self.logger.debug(f"Error parsing {osu_file.name}: {e}")
            return None
    
    def _is_suitable(self, map_info: dict) -> bool:
        """Check if map is suitable for training."""
        # Check key mode
        if map_info.get('keys') != self.key_mode:
            return False
        
        # Check difficulty (rough approximation)
        stars = map_info.get('stars', 0)
        if stars < self.min_difficulty or stars > self.max_difficulty:
            return False
        
        return True
    
    def _select_random_beatmap(self) -> Optional[dict]:
        """Select a random beatmap that wasn't played recently."""
        if not self.available_beatmaps:
            return None
        
        # Filter out recently played maps
        available = [
            bm for bm in self.available_beatmaps 
            if bm['path'] not in self.beatmap_history
        ]
        
        # If all maps were played, clear history
        if not available:
            self.beatmap_history.clear()
            available = self.available_beatmaps
        
        # Select random
        selected = random.choice(available)
        
        # Update history
        self.beatmap_history.append(selected['path'])
        if len(self.beatmap_history) > self.max_history:
            self.beatmap_history.pop(0)
        
        return selected
    
    def _change_beatmap_in_osu(self, beatmap: dict) -> bool:
        """Change to the selected beatmap in osu!."""
        try:
            self.logger.info(f"Selecting beatmap: {beatmap['name']} [{beatmap['difficulty']}]")
            
            # Press Escape to go to song select
            pydirectinput.press('escape')
            time.sleep(1.5)
            
            # Press F3 to open search
            pydirectinput.press('f3')
            time.sleep(0.3)
            
            # Clear search box
            pydirectinput.keyDown('ctrl')
            pydirectinput.press('a')
            pydirectinput.keyUp('ctrl')
            time.sleep(0.1)
            
            # Type song title
            title = beatmap['name'][:30]  # Limit length
            pydirectinput.write(title, interval=0.02)
            time.sleep(0.5)
            
            # Press Enter to select
            pydirectinput.press('enter')
            time.sleep(0.3)
            
            # Press Enter again to start
            pydirectinput.press('enter')
            time.sleep(1.0)
            
            # Final Enter to confirm (if needed)
            pydirectinput.press('enter')
            time.sleep(2.0)
            
            self.logger.info("Beatmap selection complete!")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to change beatmap: {e}")
            return False
    
    def _on_step(self) -> bool:
        """Check if song finished and select new one."""
        # Check if a song just finished
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if info.get('song_finished', False):
                    self.current_song_count += 1
                    
                    if self.verbose > 0:
                        self.logger.info(
                            f"Song #{self.current_song_count} completed! "
                            f"Selecting new random beatmap..."
                        )
                    
                    # Select and change to new beatmap
                    new_beatmap = self._select_random_beatmap()
                    
                    if new_beatmap:
                        success = self._change_beatmap_in_osu(new_beatmap)
                        if success:
                            self.logger.info(
                                f"Now playing: {new_beatmap['name']} "
                                f"[{new_beatmap['difficulty']}] "
                                f"(~{new_beatmap['stars']:.1f}★)"
                            )
                        else:
                            self.logger.warning("Failed to change beatmap, will retry next song")
                    else:
                        self.logger.warning("No suitable beatmaps available")
                    
                    # Wait a bit for osu! to load
                    time.sleep(3.0)
        
        return True


class SimpleRandomSelector(BaseCallback):
    """
    Simpler version: Just presses random arrow keys in song select.
    Less reliable but doesn't need to parse beatmap files.
    """
    
    def __init__(self, songs_per_change: int = 1, verbose: int = 1):
        """
        Initialize simple random selector.
        
        Args:
            songs_per_change: Change song after N songs
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.songs_per_change = songs_per_change
        self.songs_played = 0
        self.logger = get_logger("simple_selector")
    
    def _on_step(self) -> bool:
        """Check if should change song."""
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if info.get('song_finished', False):
                    self.songs_played += 1
                    
                    if self.songs_played >= self.songs_per_change:
                        if self.verbose > 0:
                            self.logger.info("Selecting random song...")
                        
                        self._select_random_song()
                        self.songs_played = 0
        
        return True
    
    def _select_random_song(self):
        """Navigate to random song in osu!."""
        try:
            # Go to song select
            pydirectinput.press('escape')
            time.sleep(1.0)
            
            # Press random arrow keys to navigate
            directions = ['up', 'down']
            num_presses = random.randint(3, 15)
            
            for _ in range(num_presses):
                direction = random.choice(directions)
                pydirectinput.press(direction)
                time.sleep(0.05)
            
            time.sleep(0.5)
            
            # Start the selected song
            pydirectinput.press('enter')
            time.sleep(1.0)
            pydirectinput.press('enter')
            time.sleep(2.0)
            
            self.logger.info("Random song selected!")
        
        except Exception as e:
            self.logger.error(f"Failed to select random song: {e}")

