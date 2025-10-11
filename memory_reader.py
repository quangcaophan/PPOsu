import requests
import threading
import time
from collections import deque

class FastMemoryReader:
    """
    Non-blocking memory reader using background thread
    - Background thread fetches from gosumemory continuously
    - Main thread gets cached data instantly (< 1ms)
    - No waiting for network requests
    """
    def __init__(self, url="http://127.0.0.1:24050/json", update_freq=60):
        self.url = url
        self.update_freq = update_freq  # Hz (60 = 60 times per second)
        self.update_interval = 1.0 / update_freq
        
        # Cached data
        self.cached_state = self._get_default_state()
        self.lock = threading.Lock()
        
        # Background thread
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        
        # Stats
        self.read_count = 0
        self.error_count = 0
        self.last_fetch_time = 0
        
        print("✅ FastMemoryReader started (background thread)")
    
    def _get_default_state(self):
        """Default state when no data available"""
        return {
            "game_state": 0,
            "score": 0,
            "combo": 0,
            "accuracy": 1.0,
            "miss": 0,
            "hit_50": 0,
            "hit_100": 0,
            "hit_300": 0,
            "hit_geki": 0
        }
    
    def _reader_loop(self):
        """Background thread - continuously fetch from gosumemory"""
        while self.running:
            try:
                fetch_start = time.time()
                
                # Fetch from gosumemory
                resp = requests.get(self.url, timeout=0.05)  # Very short timeout
                data = resp.json()
                
                # Parse data
                gameplay = data.get("gameplay", {})
                menu = data.get("menu", {})
                hits = gameplay.get("hits", {})
                
                new_state = {
                    "game_state": menu.get("state", 0),
                    "score": gameplay.get("score", 0),
                    "combo": gameplay.get("combo", {}).get("current", 0),
                    "accuracy": gameplay.get("accuracy", 100.0) / 100.0,
                    "miss": hits.get("0", 0),
                    "hit_50": hits.get("50", 0),
                    "hit_100": hits.get("100", 0),
                    "hit_300": hits.get("300", 0),
                    "hit_geki": hits.get("geki", 0)
                }
                
                # Update cache atomically
                with self.lock:
                    self.cached_state = new_state
                    self.read_count += 1
                
                fetch_time = (time.time() - fetch_start) * 1000
                self.last_fetch_time = fetch_time
                
            except requests.Timeout:
                # Timeout - use cached data
                with self.lock:
                    self.error_count += 1
            except (requests.RequestException, ValueError):
                # Connection error - use cached data
                with self.lock:
                    self.error_count += 1
            except Exception as e:
                with self.lock:
                    self.error_count += 1
            
            # Sleep until next update
            time.sleep(self.update_interval)
    
    def get_game_state(self):
        """
        Get cached game state (instant, ~< 1ms)
        No waiting for network request
        """
        with self.lock:
            return self.cached_state.copy()
    
    def get_stats(self):
        """Get reader statistics"""
        with self.lock:
            return {
                "read_count": self.read_count,
                "error_count": self.error_count,
                "last_fetch_time_ms": self.last_fetch_time
            }
    
    def close(self):
        """Stop background thread"""
        self.running = False
        self.reader_thread.join(timeout=2)
        print("✅ FastMemoryReader stopped")


# ============ ALTERNATIVE: Ultra-fast version with local caching ============

class UltraFastMemoryReader:
    """
    Even faster - combines background fetching with local state
    Updates happen every 16.67ms (60 Hz)
    Main thread reads in ~0.1ms
    """
    def __init__(self, url="http://127.0.0.1:24050/json"):
        self.url = url
        
        # Shared state (read by main, write by background)
        self.state = self._get_default_state()
        self.lock = threading.Lock()
        
        # Performance tracking
        self.fetch_times = deque(maxlen=100)
        self.error_count = 0
        
        # Background thread
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
    
    def _get_default_state(self):
        return {
            "game_state": 0,
            "score": 0,
            "combo": 0,
            "accuracy": 1.0,
            "miss": 0,
            "hit_50": 0,
            "hit_100": 0,
            "hit_300": 0,
            "hit_geki": 0
        }
    
    def _fetch_loop(self):
        """Background: fetch every 16.67ms (60 Hz)"""
        while self.running:
            t0 = time.time()
            
            try:
                resp = requests.get(self.url, timeout=0.01)
                data = resp.json()
                
                # Quick parse
                gameplay = data.get("gameplay", {})
                menu = data.get("menu", {})
                hits = gameplay.get("hits", {})
                
                new_state = {
                    "game_state": menu.get("state", 0),
                    "score": gameplay.get("score", 0),
                    "combo": gameplay.get("combo", {}).get("current", 0),
                    "accuracy": gameplay.get("accuracy", 100.0) / 100.0,
                    "miss": hits.get("0", 0),
                    "hit_50": hits.get("50", 0),
                    "hit_100": hits.get("100", 0),
                    "hit_300": hits.get("300", 0),
                    "hit_geki": hits.get("geki", 0)
                }
                
                with self.lock:
                    self.state = new_state
                    self.fetch_times.append(time.time() - t0)
                
            except Exception as e:
                with self.lock:
                    self.error_count += 1
            
            # Sleep for 16.67ms (60 Hz update rate)
            elapsed = time.time() - t0
            sleep_time = max(0.0167 - elapsed, 0.001)
            time.sleep(sleep_time)
    
    def get_game_state(self):
        """Get state instantly from cache"""
        with self.lock:
            return self.state.copy()
    
    def close(self):
        self.running = False
        self.thread.join(timeout=2)


# ============ Drop-in replacement ============
# Just replace in mania_env.py:
# from memory_reader import MemoryReader
# With:
# from memory_reader import FastMemoryReader as MemoryReader
# or:
# from memory_reader import UltraFastMemoryReader as MemoryReader

# Then use exactly the same:
# self.memory_reader = MemoryReader()
# game_state = self.memory_reader.get_game_state()


# ============ Test script ============
if __name__ == "__main__":
    print("Testing FastMemoryReader...")
    
    reader = FastMemoryReader(update_freq=60)
    
    print("\nReading game state 100 times...")
    times = []
    
    for i in range(100):
        t0 = time.time()
        state = reader.get_game_state()
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        
        if i % 20 == 0:
            print(f"  {i}: {elapsed:.3f}ms - State: {state['game_state']}, Score: {state['score']}")
    
    import numpy as np
    print(f"\n📊 Performance:")
    print(f"  Avg: {np.mean(times):.3f}ms")
    print(f"  Min: {np.min(times):.3f}ms")
    print(f"  Max: {np.max(times):.3f}ms")
    print(f"  FPS: {1000/np.mean(times):.1f}")
    
    stats = reader.get_stats()
    print(f"\n📈 Stats:")
    print(f"  Reads: {stats['read_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"  Last fetch: {stats['last_fetch_time_ms']:.2f}ms")
    
    reader.close()