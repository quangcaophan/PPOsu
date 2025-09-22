import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import torch
import time
from game_env import OsuManiaEnv
import json

class TrainingLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if 'episode' in self.locals['infos'][0]:
            info = self.locals['infos'][0]
            ep_info = info['episode']
            print(f"Episode Done -> Reward: {ep_info['r']:.2f}, OCR Acc: {info.get('ocr_accuracy', 0)*100:.1f}%, Combo: {info.get('current_combo', 0)}")
        return True

def load_config() -> dict:
    try:
        with open("osu_config.json", 'r') as f:
            print("✅ Configuration loaded.")
            return json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: osu_config.json not found!")
        return None

def train_agent(config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device.upper()}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True) 

    env = OsuManiaEnv(play_area=config.get('play_area'), show_window=True, config_file="osu_config.json")
    eval_env = OsuManiaEnv(play_area=config.get('play_area'), show_window=False, config_file="osu_config.json")

    template_path = "template/result_template.png"
    if os.path.exists(template_path):
        env.load_result_template(template_path)
        eval_env.load_result_template(template_path)

    model_path = "models/latest_model.zip"
    if os.path.exists(model_path):
        print(f"🔄 Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, device=device)
    else:
        print("🆕 Creating a new PPO model...")
        model = PPO("CnnPolicy", env, verbose=1, device=device, tensorboard_log="./tensorboard_logs/")

    callbacks = [
        TrainingLogCallback(),
        CheckpointCallback(
            save_freq=25000, 
            save_path="checkpoints/", 
            name_prefix="osu_checkpoint"
        ),
        EvalCallback(
            eval_env, 
            best_model_save_path='./models/best_model/', 
            log_path='./logs/', 
            eval_freq=50000, 
            n_eval_episodes=5, 
            deterministic=True,
            render=False
        )
    ]
    
    total_timesteps = 1_000_000
    print(f"\n🏁 Starting training for {total_timesteps:,} timesteps.")
    time.sleep(5)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks, 
            reset_num_timesteps=False,
            progress_bar=True
        )
        model.save("models/final_model")
    except KeyboardInterrupt:
        model.save("models/interrupted_model")
        print("\n⏹️ Training interrupted. Model saved.")
    finally:
        env.close()
        eval_env.close() 

def main():
    config = load_config()
    if config:
        train_agent(config)

if __name__ == '__main__':
    main()