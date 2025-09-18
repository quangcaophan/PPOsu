from stable_baselines3 import PPO
from game_env import OsuManiaEnv
import time
import torch

# ---- CẤU HÌNH QUAN TRỌNG ----
# Bạn cần tự xác định tọa độ vùng chơi game trên màn hình của bạn
# Có thể dùng các tool online để đo tọa độ pixel
PLAY_AREA = {'top': 230, 'left': 710, 'width': 490, 'height': 750} 
NUM_KEYS = 4


# Kiểm tra và chọn thiết bị DirectML
if torch.dml.is_available():
    device = torch.device("dml")
    print("Đang sử dụng thiết bị DirectML (AMD GPU)")
else:
    device = torch.device("cpu")
    print("Đang sử dụng CPU")


if __name__ == '__main__':
    print("Vui lòng mở osu! và vào màn hình chọn bài hát.")
    print("Script sẽ bắt đầu sau 5 giây...")
    time.sleep(5)

    # Khởi tạo môi trường game
    env = OsuManiaEnv(play_area=PLAY_AREA, num_keys=NUM_KEYS)

    # Khởi tạo Tác nhân (Agent) sử dụng thuật toán PPO
    # "CnnPolicy": Sử dụng mạng CNN có sẵn của thư viện vì đầu vào của chúng ta là ảnh.
    # verbose=1: In ra thông tin trong quá trình học.
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="./osu_ppo_tensorboard/", # Nơi lưu log để theo dõi
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='dml' # Sử dụng GPU để tăng tốc
    )

    print("Bắt đầu huấn luyện AI...")
    # Huấn luyện AI trong 1,000,000 bước
    model.learn(total_timesteps=1_000_000, progress_bar=True)

    # Lưu lại model đã huấn luyện
    model.save("ppo_osu_mania_4k")
    print("Đã lưu model!")

    env.close()