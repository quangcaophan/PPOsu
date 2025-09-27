"""
osu!mania Environment Package
Auto-selects best available environment
"""

from .constants import *

# Version info
__version__ = "2.0.0"
__author__ = "AI Assistant"

# Try to import environments in order of preference
OsuManiaEnv = None
available_environments = []

# 1. Try AsyncOsuManiaEnv (best performance)
try:
    from .mania_env_async import OsuManiaEnv as AsyncOsuManiaEnv
    OsuManiaEnv = AsyncOsuManiaEnv
    available_environments.append("AsyncOsuManiaEnv")
except ImportError:
    pass

# 2. Fallback to original
if OsuManiaEnv is None:
    try:
        from .mania_env import OsuManiaEnv as OriginalOsuManiaEnv
        OsuManiaEnv = OriginalOsuManiaEnv
        available_environments.append("OriginalOsuManiaEnv")
    except ImportError:
        pass

# Print status
if OsuManiaEnv:
    print(f"✅ Environment ready: {OsuManiaEnv.__name__}")
    if len(available_environments) > 1:
        print(f"📝 Available: {', '.join(available_environments)}")
else:
    print("❌ No osu!mania environment available!")

__all__ = ['OsuManiaEnv', 'KEY_MAPPINGS', 'FRAME_DELAY', 'TARGET_FPS', 'FRAME_SIZE']
