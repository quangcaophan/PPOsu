import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time

# Định nghĩa các phím cho chế độ 4K
KEYS = ['d', 'f', 'j', 'k']

class OsuManiaEnv(gym.Env):
    def __init__(self, play_area, num_keys=4):
        super(OsuManiaEnv, self).__init__()

        self.play_area = play_area # Tọa độ vùng chơi game, vd: {'top': 400, 'left': 100, 'width': 400, 'height': 500}
        self.sct = mss.mss()
        self.num_keys = num_keys
        
        # Không gian hành động: Mỗi hành động là một tổ hợp phím
        # Ví dụ 4K -> 2^4 = 16 hành động (từ không nhấn gì -> nhấn cả 4 phím)
        self.action_space = spaces.Discrete(2**self.num_keys)

        # Không gian trạng thái: Ảnh xám của vùng chơi game đã được xử lý
        # Chúng ta sẽ dùng 4 khung hình liên tiếp để AI cảm nhận được sự chuyển động
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4, 84, 84), dtype=np.uint8)
        
        self.last_four_frames = np.zeros((4, 84, 84), dtype=np.uint8)

    def _get_state(self):
        # Chụp ảnh màn hình vùng chơi
        sct_img = self.sct.grab(self.play_area)
        img = np.array(sct_img)

        # Tiền xử lý ảnh
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, (84, 84))
        
        return resized

    def step(self, action):
        # ---- 1. Thực hiện hành động ----
        # Giải mã 'action' (một số từ 0 -> 15) thành tổ hợp phím [True, False, True, False]
        action_combo = [bool((action >> i) & 1) for i in range(self.num_keys)]

        for i, should_press in enumerate(action_combo):
            if should_press:
                pydirectinput.keyDown(KEYS[i])
            else:
                pydirectinput.keyUp(KEYS[i])
        
        # ---- 2. Lấy trạng thái mới ----
        new_frame = self._get_state()
        self.last_four_frames = np.roll(self.last_four_frames, -1, axis=0) # Dịch chuyển các frame cũ
        self.last_four_frames[-1] = new_frame # Thêm frame mới vào cuối
        
        # ---- 3. Tính toán phần thưởng (Reward) ----
        # Đây là phần khó nhất và cần sáng tạo
        # Ví dụ đơn giản: bạn cần 1 cơ chế khác để đọc điểm số/combo/accuracy
        # Giả sử chúng ta có hàm read_score() và read_combo()
        # reward = (new_score - self.score) + (new_combo - self.combo) * 0.1
        reward = 1 # Placeholder: Thưởng 1 điểm cho mỗi bước sống sót

        # ---- 4. Kiểm tra kết thúc ----
        # Ví dụ: nếu máu về 0 hoặc bài hát kết thúc
        done = False # Placeholder

        # ---- 5. Trả về kết quả ----
        # info là một dict trống, dùng để debug
        return self.last_four_frames, reward, done, False, {}

    def reset(self, **kwargs):
        # Hàm này được gọi khi bắt đầu một màn chơi mới
        # Cần có logic để bắt đầu/restart bài hát trong osu!
        print("Resetting environment...")
        time.sleep(3) # Chờ để bắt đầu
        
        # Reset lại trạng thái ban đầu
        for _ in range(4):
            new_frame = self._get_state()
            self.last_four_frames[_] = new_frame
            time.sleep(0.05)

        return self.last_four_frames, {}

    def close(self):
        # Nhả tất cả các phím khi đóng môi trường
        for key in KEYS:
            pydirectinput.keyUp(key)