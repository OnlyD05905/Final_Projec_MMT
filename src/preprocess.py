import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import * # Import config từ utils

def load_and_preprocess():
    print("--- [GĐ1] Đang tải dữ liệu... ---")
    # 1. Load Data
    df = pd.read_csv(RAW_DATA_PATH, encoding='cp1252')
    
    # Clean tên cột (xóa khoảng trắng)
    df.columns = df.columns.str.strip()
    
    # Xử lý vô cực và NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Dữ liệu sau khi làm sạch: {df.shape}")
    
    # Fix corrupted Web Attack labels - use regex to match any corrupted character
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.replace(
        r'Web Attack .*? Brute Force', 'Web Attack – Brute Force', regex=True
    ).str.replace(
        r'Web Attack .*? Sql Injection', 'Web Attack – Sql Injection', regex=True
    ).str.replace(
        r'Web Attack .*? XSS', 'Web Attack – XSS', regex=True
    )

    # 2. Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COLUMN])
    num_classes = len(np.unique(y))
    
    # Lưu LabelEncoder để Member 3 dùng giải mã
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Đã lưu Label Encoder. Số lớp: {num_classes}")
    print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. Feature Splitting
    print("--- [GĐ1] Đang tách đặc trưng (A & B)... ---")
    X_time = df[TIME_FEATURES].values
    X_stat = df[STAT_FEATURES].values
    print(f"Input Time shape: {X_time.shape}")
    print(f"Input Stat shape: {X_stat.shape}")

    # 4. Normalization (Riêng biệt cho từng nhánh)
    scaler_time = MinMaxScaler()
    X_time = scaler_time.fit_transform(X_time)
    
    scaler_stat = MinMaxScaler()
    X_stat = scaler_stat.fit_transform(X_stat)

    # Lưu Scaler để Member 3 dùng cho data mới
    joblib.dump(scaler_time, SCALER_TIME_PATH)
    joblib.dump(scaler_stat, SCALER_STAT_PATH)

    # 5. Reshape cho LSTM (Samples, 1, Features)
    X_time = X_time.reshape(X_time.shape[0], 1, X_time.shape[1])

    # 6. Split Train/Test
    X_time_train, X_time_test, X_stat_train, X_stat_test, y_train, y_test = train_test_split(
        X_time, X_stat, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_time_train, X_stat_train, X_time_test, X_stat_test, y_train, y_test, num_classes

if __name__ == "__main__":
    # Test chạy riêng file này
    load_and_preprocess()