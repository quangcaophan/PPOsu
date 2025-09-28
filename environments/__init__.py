"""
osu!mania Environment Package - FIXED VERSION
Auto-selects best available environment
"""

from .constants import *

# Version info
__version__ = "2.0.1"
__author__ = "AI Assistant"

# Import environments with better error handling
OsuManiaEnv = None
available_environments = []
import_errors = []

print("🔧 Loading osu!mania environments...")

# 1. Try AsyncOsuManiaEnv (best performance) 
try:
    from .mania_env_async import OsuManiaEnv as AsyncOsuManiaEnv
    OsuManiaEnv = AsyncOsuManiaEnv
    available_environments.append("AsyncOsuManiaEnv")
    print("   ✅ AsyncOsuManiaEnv loaded successfully")
except ImportError as e:
    import_errors.append(f"AsyncOsuManiaEnv: {e}")
    print(f"   ❌ AsyncOsuManiaEnv failed: {e}")
except Exception as e:
    import_errors.append(f"AsyncOsuManiaEnv: {e}")
    print(f"   ❌ AsyncOsuManiaEnv error: {e}")

# 2. Try Original as fallback
if OsuManiaEnv is None:
    try:
        from .mania_env_async import OsuManiaEnv as OriginalOsuManiaEnv
        OsuManiaEnv = OriginalOsuManiaEnv
        available_environments.append("OriginalOsuManiaEnv")
        print("   ✅ OriginalOsuManiaEnv loaded as fallback")
    except ImportError as e:
        import_errors.append(f"OriginalOsuManiaEnv: {e}")
        print(f"   ❌ OriginalOsuManiaEnv failed: {e}")
    except Exception as e:
        import_errors.append(f"OriginalOsuManiaEnv: {e}")
        print(f"   ❌ OriginalOsuManiaEnv error: {e}")

# Final status
if OsuManiaEnv:
    print(f"✅ Environment ready: {OsuManiaEnv.__name__}")
    if len(available_environments) > 1:
        print(f"📝 Available: {', '.join(available_environments)}")
else:
    print("❌ CRITICAL: No osu!mania environment available!")
    print("📋 Import errors:")
    for error in import_errors:
        print(f"   - {error}")
    print("\n💡 Solutions:")
    print("   1. Check if mania_env.py or mania_env_async.py exist")
    print("   2. Run: python -c 'from environments.mania_env import OsuManiaEnv'")
    print("   3. Check for syntax errors in environment files")

__all__ = ['OsuManiaEnv', 'KEY_MAPPINGS', 'FRAME_DELAY', 'TARGET_FPS', 'FRAME_SIZE']
