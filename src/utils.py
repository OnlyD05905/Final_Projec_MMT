# src/utils.py

# --- CẤU HÌNH ĐƯỜNG DẪN ---
RAW_DATA_PATH = "data/raw/CIC-IDS2017.csv"
MODEL_PATH = "saved_models/hybrid_model.h5"
SCALER_TIME_PATH = "saved_models/scaler_time.pkl"
SCALER_STAT_PATH = "saved_models/scaler_stat.pkl"
LABEL_ENCODER_PATH = "saved_models/label_encoder.pkl"

# --- ĐỊNH NGHĨA FEATURE SPLITTING (Member 1 cần điền đúng tên cột ở đây) ---

# Input A: Nhóm Thời gian (Time) -> Cho LSTM
TIME_FEATURES = [
    'Flow Duration', 
    'Flow IAT Mean', 
    'Flow IAT Std', 
    'Flow IAT Max', 
    'Flow IAT Min',
    'Fwd IAT Total',
    'Bwd IAT Total',
    'Fwd IAT Mean',
    'Bwd IAT Mean'
]

# Input B: Nhóm Thống kê (Statistics) -> Cho DNN
STAT_FEATURES = [
    'Total Fwd Packets', 
    'Total Backward Packets', 
    'Total Length of Fwd Packets', 
    'Total Length of Bwd Packets', 
    'Fwd Packet Length Max', 
    'Fwd Packet Length Mean', 
    'Bwd Packet Length Max', 
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'SYN Flag Count', 
    'ACK Flag Count',
    'URG Flag Count'
]

# Cột nhãn
LABEL_COLUMN = "Label"