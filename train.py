import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import time
import json
import signal
import sys
from environments.mania_env import OsuManiaEnv
from environments.constants import (
    MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, TENSORBOARD_DIR,
    LATEST_MODEL_NAME, FINAL_MODEL_NAME, BEST_MODEL_DIR_NAME,
    DEFAULT_TOTAL_TIMESTEPS, DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_N_EVAL_EPISODES, DEFAULT_POLICY
)

# ============ CUSTOM CALLBACK ============
class SongFinishedEvalCallback(EvalCallback):
    """
    A custom callback that triggers an evaluation run whenever a song is finished
    and the results screen (game_state=7) is reached.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop('eval_freq', None)        
        super().__init__(*args, eval_freq=1, **kwargs)

    def _on_step(self) -> bool:
        # Check if an episode has ended
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            
            if info.get("game_state") == 7:
                if self.verbose > 0:
                    print("\n🎶 Song finished! Triggering evaluation...")
                    time.sleep(5)
                
                # Trick the parent class to run evaluation.
                # We save and restore n_calls to not affect other callbacks.
                original_n_calls = self.n_calls
                self.n_calls = self.eval_freq  # This is 1
                continue_training = super()._on_step()
                
                if self.verbose > 0:
                    print("🏁 Evaluation complete! Resuming training.")

                self.n_calls = original_n_calls
                
                return continue_training
        
        return True

# ============ TRAINING MANAGER ============

class TrainingManager:
    """Complete training management with cleanup"""
    
    def __init__(self, config_path: str, show_eval_window: bool = False):
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.env = None
        self.eval_env = None
        self.show_eval_window = show_eval_window
        
        # Setup paths
        self.setup_paths()
        self.setup_signal_handlers()
    
    def _load_config(self):
        """Load and validate config"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            required_fields = ['mode', 'play_area', 'num_keys']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field: {field}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    def setup_paths(self):
        """Setup directory structure"""
        mode = self.config.get('mode', 'mania')
        keys = self.config.get('num_keys')
        
        self.run_id = f"{mode}" + (f"_{keys}k" if keys else "")
        
        # Directories
        key_str = f"{keys}k" if keys else ""
        dirs = {
            'model': os.path.join(MODELS_DIR, mode, key_str),
            'log': os.path.join(LOGS_DIR, self.run_id),
            'checkpoint': os.path.join(CHECKPOINTS_DIR, self.run_id),
            'tensorboard': TENSORBOARD_DIR
        }
        
        for name, path in dirs.items():
            os.makedirs(path, exist_ok=True)
            setattr(self, f"{name}_dir", path)
        
        # File paths
        self.latest_model_path = os.path.join(self.model_dir, LATEST_MODEL_NAME)
        self.final_model_path = os.path.join(self.model_dir, FINAL_MODEL_NAME)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n⏹️ Interrupted by signal {signum}")
            self.cleanup_and_save("Signal interrupt")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def create_environments(self):
        """Create training environments"""
        print(f"🏗️ Creating environments...")
        
        # Training environment: no auto-restart
        self.env = OsuManiaEnv(
            config_path=self.config_path, 
            show_window=True,
            run_id=self.run_id,
            is_eval_env=False
        )
        self.env = Monitor(self.env)

        # Evaluation environment: will auto-restart songs
        self.eval_env = OsuManiaEnv(
            config_path=self.config_path, 
            show_window=self.show_eval_window,
            is_eval_env=True
        )
        self.eval_env = Monitor(self.eval_env)
    
    def create_model(self):
        """Create or load PPO model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Device: {device.upper()}")

        training_params = self.config.get("training_params", {})
        ppo_params = training_params.get("ppo_params", {})
        
        if os.path.exists(self.latest_model_path):
            print(f"🔄 Loading: {self.latest_model_path}")
            temp_model = PPO.load(self.latest_model_path, device=device)
            if temp_model.observation_space != self.env.observation_space:
                print(f"❌ Observation space mismatch!\n"
                      f"   - Model was saved with: {temp_model.observation_space}\n"
                      f"   - Current environment has: {self.env.observation_space}\n"
                      f"   This is likely because you changed FRAME_SIZE or observation preprocessing.\n"
                      f"   👉 Please delete the old model file to start fresh:\n"
                      f"   {self.latest_model_path}")
                sys.exit(1)
            self.model = PPO.load(self.latest_model_path, env=self.env, device=device)
        else:
            print("🆕 Creating new PPO model...")
            self.model = PPO(
                policy=training_params.get("policy", DEFAULT_POLICY),
                env=self.env,
                verbose=1,
                device=device,
                tensorboard_log=self.tensorboard_dir,
                **ppo_params
            )
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        training_params = self.config.get("training_params", {})
        callback_params = training_params.get("callback_params", {})
        self.callbacks = [
            CheckpointCallback(
                save_freq=callback_params.get("checkpoint_save_freq", DEFAULT_CHECKPOINT_FREQ),
                name_prefix=self.run_id,
                save_path=self.model_dir
            ),
            SongFinishedEvalCallback(
                self.eval_env,
                best_model_save_path=os.path.join(self.model_dir, BEST_MODEL_DIR_NAME),
                n_eval_episodes=callback_params.get("n_eval_episodes", DEFAULT_N_EVAL_EPISODES)
            )
        ]
    
    def train(self, total_timesteps):
        """Run training"""
        print(f"\n🏁 Training {total_timesteps:,} timesteps")
        print(f"📊 ID: {self.run_id}")
        print("⚠️  Ctrl+C for safe exit")
        time.sleep(3)
        
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                reset_num_timesteps=(not os.path.exists(self.latest_model_path)),
                progress_bar=True,
                tb_log_name=self.run_id
            )
            
            print("🎉 Training completed!")
            self.model.save(self.final_model_path)
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            
        finally:
            hours = (time.time() - start_time) / 3600
            self.cleanup_and_save(f"Training time: {hours:.1f}h")
    
    def cleanup_and_save(self, reason=""):
        """Cleanup and save"""
        print(f"\n🧹 Cleanup: {reason}")
        
        if self.model:
            try:
                self.model.save(self.latest_model_path)
                print(f"✅ Saved: {self.latest_model_path}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
        
        # Close environments
        for env_name in ['env', 'eval_env']:
            env = getattr(self, env_name, None)
            if env:
                try:
                    env.close()
                except Exception:
                    pass
        
        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Complete osu!mania AI Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps from config")
    parser.add_argument("--show-eval", action="store_true", help="Show the evaluation environment window.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    try:
        trainer = TrainingManager(args.config, show_eval_window=args.show_eval)
        
        # Determine timesteps: CLI > config > default
        timesteps = args.timesteps
        if timesteps is None:
            timesteps = trainer.config.get("training_params", {}).get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)

        trainer.create_environments()
        trainer.create_model()
        trainer.setup_callbacks()
        trainer.train(timesteps)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()