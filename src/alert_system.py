import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.utils import *

class AlertSystem:
    def __init__(self):
        print("--- [GĐ4] Khởi động hệ thống cảnh báo... ---")
        # Load các thành phần cần thiết
        self.model = load_model(MODEL_PATH)
        self.scaler_time = joblib.load(SCALER_TIME_PATH)
        self.scaler_stat = joblib.load(SCALER_STAT_PATH)
        self.le = joblib.load(LABEL_ENCODER_PATH)
        print("Hệ thống đã sẵn sàng!")

    def predict_and_alert(self, raw_time_data, raw_stat_data):
        # 1. Tiền xử lý dữ liệu mới (Giống hệt GĐ1)
        # Scale
        processed_time = self.scaler_time.transform(raw_time_data)
        processed_stat = self.scaler_stat.transform(raw_stat_data)
        
        # Reshape cho LSTM
        processed_time = processed_time.reshape(processed_time.shape[0], 1, processed_time.shape[1])

        # 2. Dự đoán
        probs = self.model.predict([processed_time, processed_stat], verbose=0)
        
        # 3. Phân tích kết quả
        for i, prob in enumerate(probs):
            risk_score = np.max(prob)
            class_idx = np.argmax(prob)
            attack_name = self.le.inverse_transform([class_idx])[0]

            self._trigger_alert(attack_name, risk_score)

    def _trigger_alert(self, attack_name, score):
        # Logic cảnh báo
        if attack_name == "BENIGN":
            print(f"✅ Normal Traffic (Score: {score:.2f})")
        else:
            if score > 0.9:
                print(f"🚨 [CRITICAL] Phát hiện: {attack_name} | Risk: {score:.2f} -> BLOCK IP!")
            elif score > 0.7:
                print(f"⚠️ [WARNING] Nghi ngờ: {attack_name} | Risk: {score:.2f} -> Ghi log.")
            else:
                print(f"ℹ️ [INFO] Có thể là: {attack_name} | Risk: {score:.2f}")

# --- GIẢ LẬP CHẠY THỬ ---
if __name__ == "__main__":
    bot = AlertSystem()
    
    # Giả sử có dữ liệu mới (raw)
    # Member 3 cần đảm bảo số lượng cột khớp với TIME_FEATURES và STAT_FEATURES trong utils.py
    # Đây là dữ liệu giả (random) để test code chạy
    dummy_time = np.random.rand(5, len(TIME_FEATURES)) * 1000 
    dummy_stat = np.random.rand(5, len(STAT_FEATURES)) * 100

    bot.predict_and_alert(dummy_time, dummy_stat)