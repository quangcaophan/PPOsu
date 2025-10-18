"""
Quick setup script for PPO Osu! Mania agent.
This is a simple wrapper around the main setup tool.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the setup tool."""
    print("🎮 PPO Osu! Mania Agent Setup")
    print("=" * 40)
    
    # Check if setup_tool.py exists
    setup_tool_path = Path(__file__).parent / "setup_tool.py"
    if not setup_tool_path.exists():
        print("❌ Setup tool not found!")
        print("Please make sure setup_tool.py is in the same directory.")
        return
    
    try:
        # Run the setup tool
        subprocess.run([sys.executable, str(setup_tool_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed with error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Setup cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
