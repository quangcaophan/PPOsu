import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import torch
import time
import json
import signal
import sys
from environments import OsuManiaEnv
# Try to import performance kd
try:
    from performance_profiler import profiler, start_profiling, stop_profiling
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

class AutoSaveCallback(BaseCallback):
    """Auto-save model every n_steps and every 5 minutes"""
    def __init__(self, save_path: str, save_freq: int = 512, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_counter = 0
        self.last_save_time = time.time()

    def _on_step(self) -> bool:
        self.save_counter += 1
        current_time = time.time()
        
        # Save conditions: step count OR time threshold
        step_condition = self.save_counter >= self.save_freq
        time_condition = current_time - self.last_save_time > 300  # 5 minutes
        
        if step_condition or time_condition:
            if self.verbose > 0:
                reason = "steps" if step_condition else "time"
                print(f"💾 Auto-saving model at step {self.num_timesteps} ({reason})")
            
            try:
                self.model.save(self.save_path)
                self.save_counter = 0
                self.last_save_time = current_time
                if self.verbose > 0:
                    print(f"✅ Model saved to {self.save_path}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
                
        return True

class TrainingManager:
    """Complete training management with cleanup"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.env = None
        self.eval_env = None
        
        # Setup paths
        self.setup_paths()
        self.setup_signal_handlers()
        
        # Start profiling
        if PROFILING_AVAILABLE:
            start_profiling()
    
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
        dirs = {
            'model': f"models/{mode}/" + (f"{keys}k/" if keys else ""),
            'log': f"logs/{self.run_id}",
            'checkpoint': f"checkpoints/{self.run_id}",
            'tensorboard': "tensorboard_logs/",
            'template': f"templates/{self.run_id}"
        }
        
        for name, path in dirs.items():
            os.makedirs(path, exist_ok=True)
            setattr(self, f"{name}_dir", path)
        
        # File paths
        self.latest_model_path = f"{self.model_dir}/latest_model.zip"
        self.final_model_path = f"{self.model_dir}/final_model.zip"
        self.template_path = f"{self.template_dir}/result_template.png"
    
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
        try:
            from environments import OsuManiaEnv
        except ImportError:
            raise ImportError("Could not import OsuManiaEnv. Run migration first!")
        
        print(f"🏗️ Creating environments...")
        
        self.env = OsuManiaEnv(
            config_path=self.config_path, 
            show_window=True,
            run_id=self.run_id,
            log_dir=self.log_dir
        )
        self.eval_env = OsuManiaEnv(
            config_path=self.config_path, 
            show_window=False
        )
        
        # Load templates
        if os.path.exists(self.template_path):
            self.env.load_result_template(self.template_path)
            self.eval_env.load_result_template(self.template_path)
            print(f"✅ Template loaded: {self.template_path}")
    
    def create_model(self):
        """Create or load PPO model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Device: {device.upper()}")
        
        if os.path.exists(self.latest_model_path):
            print(f"🔄 Loading: {self.latest_model_path}")
            self.model = PPO.load(self.latest_model_path, env=self.env, device=device)
        else:
            print("🆕 Creating new PPO model...")
            self.model = PPO(
                policy="CnnPolicy",
                env=self.env,
                verbose=1,
                device=device,
                tensorboard_log=self.tensorboard_dir,
                n_steps=512,
                batch_size=32,
                ent_coef=0.01,
                learning_rate=2.5e-4,
                clip_range=0.2,
                n_epochs=4
            )
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        self.callbacks = [
            AutoSaveCallback(
                save_path=self.latest_model_path,
                save_freq=512,
                verbose=1
            ),
            CheckpointCallback(
                save_freq=10000,
                save_path=self.checkpoint_dir,
                name_prefix=self.run_id
            ),
            EvalCallback(
                self.eval_env,
                best_model_save_path=f"{self.model_dir}/best_model/",
                log_path=self.log_dir,
                eval_freq=5000,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
        ]
    
    def train(self, total_timesteps=100_000):
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
        
        # Stop profiling
        if PROFILING_AVAILABLE:
            try:
                stop_profiling()
            except Exception:
                pass
        
        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description="Complete osu!mania AI Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)
    
    try:
        trainer = TrainingManager(args.config)
        trainer.create_environments()
        trainer.create_model()
        trainer.setup_callbacks()
        trainer.train(args.timesteps)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()


# Add these imports to train_optimized.py if not already present
import logging
from stable_baselines3.common.logger import configure

# Add this class after other callbacks
class OCRMonitorCallback(BaseCallback):
    """Monitor OCR performance during training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ocr_stats_log = []
    
    def _on_step(self) -> bool:
        # Log OCR stats every 1000 steps
        if self.num_timesteps % 1000 == 0:
            info = self.locals.get('infos', [{}])
            if info and len(info) > 0:
                ocr_stats = info[0].get('ocr_stats')
                if ocr_stats:
                    for area, stats in ocr_stats.items():
                        self.logger.record(f"ocr/{area}_success_rate", stats['success_rate'])
                        self.logger.record(f"ocr/{area}_attempts", stats['attempts'])
                
                # Also log current values
                combo = info[0].get('current_combo', 0)
                score = info[0].get('current_score', 0)
                accuracy = info[0].get('ocr_accuracy', 0)
                
                self.logger.record("env/combo", combo)
                self.logger.record("env/score", score) 
                self.logger.record("env/accuracy", accuracy)
        
        return True

# Usage in TrainingManager.setup_callbacks():
# self.callbacks.append(OCRMonitorCallback(verbose=1))
