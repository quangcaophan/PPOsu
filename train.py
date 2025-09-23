import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
import time
import json

def get_env_class(mode_name):
    if mode_name == 'mania':
        from environments.mania_env import OsuManiaEnv
        return OsuManiaEnv
    # elif mode_name == 'taiko':
    #     from environments.taiko_env import OsuTaikoEnv
    #     return OsuTaikoEnv
    else:
        raise ValueError(f"Unknown game mode: {mode_name}")

def train_agent(config_path: str):
    """
    Hàm train agent chính, hoạt động dựa trên file config được cung cấp.
    """
    # 1. Tải cấu hình
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Configuration file not found at '{config_path}'")
        return

    mode_name = config.get('mode')
    key_mode = config.get('num_keys')
    
    # Tạo một định danh duy nhất cho lần chạy này (ví dụ: 'mania_4k')
    run_id = f"{mode_name}" + (f"_{key_mode}k" if key_mode else "")
    print(f"--- Starting training session for: {run_id} ---")

    # 2. Thiết lập các đường dẫn một cách tự động
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device.upper()}")
    
    model_dir = f"models/{mode_name}/" + (f"{key_mode}k/" if key_mode else "")
    log_dir = f"logs/{run_id}"
    checkpoint_dir = f"checkpoints/{run_id}"
    tensorboard_log_dir = f"tensorboard_logs/"
    template_dir = f"templates/{run_id}"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # 3. Tải và khởi tạo môi trường tương ứng
    EnvClass = get_env_class(mode_name)
    env = EnvClass(config_path=config_path, show_window=True)
    eval_env = EnvClass(config_path=config_path, show_window=False)

    template_path = f"{template_dir}/result_template.png"
    if os.path.exists(template_path):
        env.load_result_template(template_path)
        eval_env.load_result_template(template_path)

    # 4. Tạo hoặc tải model
    latest_model_path = f"{model_dir}/latest_model.zip"
    if os.path.exists(latest_model_path):
        print(f"🔄 Loading existing model from {latest_model_path}")
        model = PPO.load(latest_model_path, env=env, device=device)
    else:
        print("🆕 Creating a new PPO model...")
        model = PPO("CnnPolicy", env, verbose=1, device=device, tensorboard_log=tensorboard_log_dir)

    # 5. Cấu hình Callbacks với các đường dẫn tự động
    callbacks = [
        CheckpointCallback(
            save_freq=25000, 
            save_path=checkpoint_dir, 
            name_prefix=run_id
        ),
        EvalCallback(
            eval_env, 
            best_model_save_path=f"{model_dir}/best_model/", 
            log_path=log_dir, 
            eval_freq=50000, 
            n_eval_episodes=5, 
            deterministic=True,
            render=False
        )
    ]
    
    # 6. Bắt đầu huấn luyện
    total_timesteps = 1_000_000
    print(f"\n🏁 Starting training for {total_timesteps:,} timesteps.")
    time.sleep(5)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks, 
            reset_num_timesteps=(not os.path.exists(latest_model_path)), # Reset nếu là model mới
            progress_bar=True,
            tb_log_name=run_id
        )
        model.save(f"{model_dir}/final_model.zip")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
    finally:
        model.save(latest_model_path)
        print(f"🔄 Latest progress saved to {latest_model_path}")
        env.close()
        eval_env.close() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an PPO agent for osu!")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file (e.g., config/mania_4k_config.json)"
    )
    args = parser.parse_args()
    
    train_agent(config_path=args.config)