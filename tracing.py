import json
import time
import os
from datetime import datetime

class JSONTracer:
    """
    Ghi lại các bước của môi trường vào file JSON để debug và phân tích chi tiết.
    Mỗi episode sẽ được lưu vào một file riêng.
    """
    def __init__(self, log_dir: str, run_id: str):
        self.log_dir = os.path.join(log_dir, "traces")
        self.run_id = run_id
        self.episode_count = 0
        self.log_file = None
        self.file_handler = None
        os.makedirs(self.log_dir, exist_ok=True)

    def start_episode(self):
        """Đóng file log cũ và mở file mới cho episode mới."""
        self.close()  # Đảm bảo file cũ đã được đóng
        self.episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.run_id}_ep{self.episode_count}_{timestamp}.jsonl"
        self.log_file = os.path.join(self.log_dir, filename)
        
        try:
            self.file_handler = open(self.log_file, 'w', encoding='utf-8')
        except IOError as e:
            print(f"❌ Lỗi khi mở file JSON trace: {e}")
            self.file_handler = None

    def log_step(self, data: dict):
        """Ghi một dictionary dữ liệu của step thành một dòng mới trong file JSONL."""
        if not self.file_handler:
            return
            
        try:
            # Sử dụng định dạng JSONL (mỗi object JSON trên một dòng) cho hiệu quả
            json.dump(data, self.file_handler)
            self.file_handler.write('\n')
        except (IOError, TypeError) as e:
            print(f"❌ Lỗi khi ghi file JSON trace: {e}")

    def close(self):
        """Đóng file log hiện tại."""
        if self.file_handler:
            self.file_handler.close()
            self.file_handler = None