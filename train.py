"""
Enhanced Training Script for Osu! Mania RL Agent
Features:
- Automatic configuration loading
- OCR-based reward system  
- Long note support
- Advanced monitoring and callbacks
- Performance optimization
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
import torch
import time
import numpy as np
from collections import deque
import json
from typing import Dict, Any
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Try to import DirectML for AMD GPUs
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("DirectML available. Using DirectML.")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("DirectML not available. Using CUDA/CPU.")

if DIRECTML_AVAILABLE:
    device = torch_directml.device()
    print(f"✅ Using device: DirectML ({torch_directml.device_name(0)})")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ Using device: CUDA ({torch.cuda.device_name(0)})")
else:
    device = torch.device('cpu')
    print("✅ Using device: CPU")

# Import our enhanced environment
from game_env import OsuManiaEnv

class TrainingCallback(BaseCallback):
    """Advanced callback for monitoring training progress with OCR metrics"""
    
    def __init__(self, verbose=0, save_freq=10000):
        super().__init__(verbose)
        self.save_freq = save_freq
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_accuracies = deque(maxlen=100)
        self.episode_ocr_accuracies = deque(maxlen=100)
        self.episode_combos = deque(maxlen=100)
        
        # Performance tracking
        self.best_reward = -float('inf')
        self.best_accuracy = 0.0
        self.best_combo = 0
        
        # Training metrics
        self.total_steps = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        self.total_steps += 1
        
        # Log episode statistics when available
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Extract osu-specific metrics from the last step
                current_info = self.locals.get('infos', [{}])[0]
                accuracy = current_info.get('accuracy', 0) * 100
                ocr_accuracy = current_info.get('ocr_accuracy', 0) * 100
                max_combo = current_info.get('current_combo', 0)
                
                self.episode_accuracies.append(accuracy)
                self.episode_ocr_accuracies.append(ocr_accuracy)
                self.episode_combos.append(max_combo)
                
                # Log to tensorboard/logger
                self.logger.record('episode/reward', episode_reward)
                self.logger.record('episode/length', episode_length)
                self.logger.record('episode/accuracy', accuracy)
                self.logger.record('episode/ocr_accuracy', ocr_accuracy)
                self.logger.record('episode/max_combo', max_combo)
                
                # Rolling averages
                if len(self.episode_rewards) >= 10:
                    self.logger.record('episode/reward_mean_10', np.mean(list(self.episode_rewards)[-10:]))
                    self.logger.record('episode/accuracy_mean_10', np.mean(list(self.episode_accuracies)[-10:]))
                    self.logger.record('episode/ocr_accuracy_mean_10', np.mean(list(self.episode_ocr_accuracies)[-10:]))
                    self.logger.record('episode/combo_mean_10', np.mean(list(self.episode_combos)[-10:]))
                
                if len(self.episode_rewards) >= 100:
                    self.logger.record('episode/reward_mean_100', np.mean(self.episode_rewards))
                    self.logger.record('episode/accuracy_mean_100', np.mean(self.episode_accuracies))
                    self.logger.record('episode/ocr_accuracy_mean_100', np.mean(self.episode_ocr_accuracies))
                    self.logger.record('episode/combo_mean_100', np.mean(self.episode_combos))
                
                # Check for new records
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.model.save("models/best_reward_model")
                    print(f"🏆 New best reward: {episode_reward:.2f}")
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.model.save("models/best_accuracy_model")
                    print(f"[TARGET] New best accuracy: {accuracy:.1f}%")
                
                if max_combo > self.best_combo:
                    self.best_combo = max_combo
                    self.model.save("models/best_combo_model")
                    print(f"🔥 New best combo: {max_combo}")
                
                # Detailed episode summary
                elapsed = time.time() - self.start_time
                print(f"Episode completed | Reward: {episode_reward:6.1f} | "
                      f"Calc Acc: {accuracy:5.1f}% | OCR Acc: {ocr_accuracy:5.1f}% | Combo: {max_combo:3d} | "
                      f"Length: {episode_length:4d} | Time: {elapsed/60:.1f}m")
        
        # Log training progress every save_freq steps
        if self.total_steps % self.save_freq == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.total_steps / elapsed
            
            self.logger.record('training/steps_per_second', steps_per_sec)
            self.logger.record('training/elapsed_minutes', elapsed / 60)
            
            print(f"Training progress: {self.total_steps:,} steps | "
                  f"{steps_per_sec:.1f} steps/sec | "
                  f"{elapsed/60:.1f} minutes elapsed")
        
        return True

class EvaluationCallback(BaseCallback):
    """Callback for periodic evaluation during training"""
    
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=3, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -float('inf')
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n[TEST] Running evaluation at step {self.n_calls}...")
            
            episode_rewards = []
            episode_accuracies = []
            episode_combos = []
            
            for i in range(self.n_eval_episodes):
                print(f"Evaluation episode {i+1}/{self.n_eval_episodes}")
                
                obs, info = self.eval_env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                max_steps = 2000  # Limit eval episodes
                
                while not done and step_count < max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    step_count += 1
                
                accuracy = info.get('accuracy', 0) * 100
                combo = info.get('current_combo', 0)
                
                episode_rewards.append(episode_reward)
                episode_accuracies.append(accuracy)
                episode_combos.append(combo)
                
                print(f"  Episode {i+1}: Reward={episode_reward:.1f}, "
                      f"Accuracy={accuracy:.1f}%, Combo={combo}")
            
            mean_reward = np.mean(episode_rewards)
            mean_accuracy = np.mean(episode_accuracies)
            mean_combo = np.mean(episode_combos)
            
            # Log evaluation results
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_accuracy', mean_accuracy)
            self.logger.record('eval/mean_combo', mean_combo)
            
            print(f"[CHART] Evaluation results:")
            print(f"   Mean Reward: {mean_reward:.1f}")
            print(f"   Mean Accuracy: {mean_accuracy:.1f}%") 
            print(f"   Mean Combo: {mean_combo:.0f}")
            
            # Save best evaluation model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("models/best_eval_model")
                print(f"[SAVE] New best evaluation model saved (reward: {mean_reward:.1f})")
        
        return True

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_file = "osu_config.json"
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"[OK] Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] Configuration file {config_file} not found!")
        print("Please run the setup tool first: python enhanced_setup_tool.py")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {config_file}")
        return None

def setup_device() -> str:
    """Setup and return the best available device"""
    if DIRECTML_AVAILABLE and torch_directml.is_available():
        device = torch_directml.device()
        print("🔥 Using DirectML (AMD GPU acceleration)")
        return device
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"[ROCKET] Using CUDA GPU: {torch.cuda.get_device_name()}")
        return device
    else:
        device = "cpu"
        print("💻 Using CPU (consider getting a GPU for faster training)")
        return device

def setup_directories():
    """Create necessary directories for training"""
    directories = ["models", "logs", "tensorboard_logs", "checkpoints", "evaluation_logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Training directories created")

def create_model(env, device: str, config: Dict[str, Any]) -> PPO:
    """Create PPO model with optimized hyperparameters for osu! mania"""
    
    # Model configuration optimized for rhythm games
    model_config = {
        "policy": "CnnPolicy",
        "env": env,
        "verbose": 1,
        "device": device,
        
        # Core hyperparameters optimized for osu!
        "n_steps": 2048,  # Collect more experience before updates
        "batch_size": 64,  # Smaller batches for stability
        "n_epochs": 10,   # Multiple epochs per update
        "learning_rate": 3e-4,  # Conservative learning rate
        
        # Advanced parameters
        "gamma": 0.99,    # Discount factor - value future rewards highly
        "gae_lambda": 0.95,  # GAE parameter for advantage estimation
        "clip_range": 0.2,   # PPO clipping parameter
        "clip_range_vf": None,  # No value function clipping
        
        # Regularization
        "ent_coef": 0.01,    # Entropy coefficient for exploration
        "vf_coef": 0.5,      # Value function coefficient
        "max_grad_norm": 0.5,  # Gradient clipping
        
        # Logging
        "tensorboard_log": "tensorboard_logs/",
    }
    
    # Load existing model if available
    latest_model_path = "models/latest_model.zip"
    if os.path.exists(latest_model_path):
        print(f"🔄 Loading existing model from {latest_model_path}")
        
        try:
            # Load model and verify compatibility
            model = PPO.load(latest_model_path, env=env, device=device)
            
            # Test compatibility
            test_obs = env.observation_space.sample()
            model.predict(test_obs)
            
            print("✅ Model compatibility verified")
            print("[OK] Existing model loaded successfully")
            return model
            
        except Exception as e:
            print(f"⚠️ Could not load existing model: {e}")
            print("Creating new model instead...")
    
    print("🆕 Creating new PPO model...")
    model = PPO(**model_config)
    return model

def train_agent(config: Dict[str, Any]):
    """Main training function"""
    
    # Setup
    device = setup_device()
    setup_directories()
    
    # Create environment
    print("🎮 Creating training environment...")
    
    # Extract areas from config
    play_area = config.get('play_area')
    if not play_area:
        print("[ERROR] Play area not found in config!")
        return False
    
    env = OsuManiaEnv(
        play_area=play_area,
        num_keys=4,
        show_window=True,  # Show visualization during training
        config_file="osu_config.json"
    )
    
    # Load result template if available
    if os.path.exists("result_template.png"):
        env.load_result_template("result_template.png")
        print("📸 Result screen template loaded")
    
    # Create evaluation environment (without visualization for speed)
    print("[CHART] Creating evaluation environment...")
    eval_env = OsuManiaEnv(
        play_area=play_area,
        num_keys=4,
        show_window=False,  # No visualization for evaluation
        config_file="osu_config.json"
    )
    
    # Create model
    model = create_model(env, device, config)
    
    # Setup custom logger
    logger = configure("logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Setup callbacks
    callbacks = []
    
    # Main training callback
    training_callback = TrainingCallback(verbose=1, save_freq=10000)
    callbacks.append(training_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="checkpoints/",
        name_prefix="osu_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvaluationCallback(
        eval_env=eval_env,
        eval_freq=50000,
        n_eval_episodes=3,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Training configuration
    total_timesteps = 1_000_000  # 1M steps for thorough training
    save_interval = 100_000      # Save every 100k steps
    
    print(f"[ROCKET] Starting training for {total_timesteps:,} timesteps")
    print("Training will be saved automatically. You can stop anytime with Ctrl+C")
    print("\nWait 10 seconds then start playing osu! mania...")
    print("Tips for training:")
    print("- Start with easier songs")
    print("- Let it play multiple songs")
    print("- Monitor the visualization window")
    print("- Check tensorboard for detailed metrics")
    
    time.sleep(10)
    
    try:
        # Training loop with progressive saves
        for i in range(0, total_timesteps, save_interval):
            remaining_steps = min(save_interval, total_timesteps - i)
            
            print(f"\n{'='*60}")
            print(f"🏃 Training segment: steps {i:,} to {i + remaining_steps:,}")
            print(f"{'='*60}")
            
            # Train for this segment
            model.learn(
                total_timesteps=remaining_steps,
                callback=callbacks,
                reset_num_timesteps=False,  # Keep step counter continuous
                progress_bar=True,
                tb_log_name="osu_training"
            )
            
            # Save progress
            model.save(f"models/model_step_{i + remaining_steps:06d}")
            model.save("models/latest_model")
            
            print(f"[SAVE] Model saved at step {i + remaining_steps:,}")
            
            # Quick performance summary
            if len(training_callback.episode_rewards) > 0:
                recent_rewards = list(training_callback.episode_rewards)[-10:]
                recent_accuracies = list(training_callback.episode_accuracies)[-10:]
                
                print(f"📈 Recent performance (last 10 episodes):")
                print(f"   Average Reward: {np.mean(recent_rewards):.1f}")
                print(f"   Average Accuracy: {np.mean(recent_accuracies):.1f}%")
                print(f"   Best Records - Reward: {training_callback.best_reward:.1f}, "
                      f"Accuracy: {training_callback.best_accuracy:.1f}%, "
                      f"Combo: {training_callback.best_combo}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        model.save("models/interrupted_model")
        print("[SAVE] Model saved as 'interrupted_model'")
    
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        model.save("models/error_backup_model")
        print("[SAVE] Backup model saved")
        raise
    
    finally:
        # Final save and cleanup
        model.save("models/final_model")
        print("[SAVE] Final model saved")
        
        # Final evaluation
        print("\n[TEST] Running final evaluation...")
        try:
            eval_callback._on_step()  # Trigger final evaluation
        except:
            print("Could not run final evaluation")
        
        # Cleanup
        env.close()
        eval_env.close()
        
        # Training summary
        print(f"\n{'='*60}")
        print("🏁 TRAINING COMPLETED!")
        print(f"{'='*60}")
        
        if len(training_callback.episode_rewards) > 0:
            print(f"[CHART] Final Statistics:")
            print(f"   Total Episodes: {len(training_callback.episode_rewards)}")
            print(f"   Best Reward: {training_callback.best_reward:.1f}")
            print(f"   Best Accuracy: {training_callback.best_accuracy:.1f}%")
            print(f"   Best Combo: {training_callback.best_combo}")
            print(f"   Average Recent Reward: {np.mean(list(training_callback.episode_rewards)[-20:]):.1f}")
        
        print(f"[SAVE] Models saved in 'models/' directory")
        print(f"📈 Training logs in 'logs/' and 'tensorboard_logs/'")
        print(f"[ROCKET] Use 'python play_agent.py' to test your trained agent!")

def test_configuration(config: Dict[str, Any]) -> bool:
    """Test configuration before starting training"""
    print("[TEST] Testing configuration...")
    
    try:
        play_area = config.get('play_area')
        env = OsuManiaEnv(
            play_area=play_area,
            num_keys=4,
            show_window=True,
            config_file="osu_config.json"
        )
        
        print("Testing environment for 10 seconds...")
        print("Make sure osu! is running and you can see the visualization window")
        
        obs, info = env.reset()
        
        for i in range(60):  # 10 seconds at ~60 FPS
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.016)
            
            if i % 30 == 0:
                print(f"Step {i}: Combo={info.get('current_combo', 0)}, "
                      f"Accuracy={info.get('accuracy', 0)*100:.1f}%")
        
        env.close()
        print("[OK] Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def main():
    print("="*60)
    print("🎵 ENHANCED OSU! MANIA RL TRAINING SCRIPT")
    print("="*60)
    
    # Load configuration
    config = load_config()
    if not config:
        print("\n🛠️ Please run setup first:")
        print("   python enhanced_setup_tool.py")
        return
    
    # Verify required dependencies
    try:
        import easyocr
        print("[OK] EasyOCR available for text recognition")
    except ImportError:
        print("[ERROR] EasyOCR not found! Install with: pip install easyocr")
        return
    
    # Show configuration
    print(f"\n⚙️ Configuration loaded:")
    print(f"   Play area: {config['play_area']}")
    print(f"   Combo area: {config.get('combo_area', 'Not set')}")
    print(f"   Score area: {config.get('score_area', 'Not set')}")
    
    # Test configuration option
    test_config = input("\n[TEST] Test configuration before training? (y/n): ").lower().strip()
    if test_config == 'y':
        if not test_configuration(config):
            print("Please fix configuration issues before training")
            return
    
    # Start training
    start_training = input("\n[ROCKET] Start training? (y/n): ").lower().strip()
    if start_training == 'y':
        train_agent(config)
    else:
        print("Training cancelled")

if __name__ == '__main__':
    # Handle command line arguments
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Just test configuration
            config = load_config()
            if config:
                test_configuration(config)
        
        elif command == "eval":
            # Run evaluation only
            print("Evaluation mode not implemented yet")
            
        else:
            print("Usage: python enhanced_train.py [test|eval]")
    else:
        main()