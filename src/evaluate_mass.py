import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from src.utils import *
import os

# Tắt log rác của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate_mass_attack():
    print("⏳ ĐANG TẢI DỮ LIỆU TỔNG (Khoảng 2.8 triệu dòng)...")
    df = pd.read_csv(RAW_DATA_PATH, encoding='cp1252')
    
    # 1. Sửa lỗi tên cột (quan trọng)
    df.columns = df.columns.str.strip()
    print("🧹 Đang quét dọn dữ liệu rác (Infinity/NaN)...")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Fix corrupted Web Attack labels - use regex to match any corrupted character
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.replace(
        r'Web Attack .*? Brute Force', 'Web Attack – Brute Force', regex=True
    ).str.replace(
        r'Web Attack .*? Sql Injection', 'Web Attack – Sql Injection', regex=True
    ).str.replace(
        r'Web Attack .*? XSS', 'Web Attack – XSS', regex=True
    )
    
    # 2. Chỉ lấy dữ liệu TẤN CÔNG (Bỏ qua BENIGN để test khả năng bắt trộm)
    # Nếu bạn muốn test cả BENIGN thì bỏ dòng này đi
    attack_df = df[df[LABEL_COLUMN] != 'BENIGN']
    
    total_attacks = len(attack_df)
    print(f"✅ Tìm thấy tổng cộng: {total_attacks} mẫu tấn công trong kho dữ liệu.")
    
    # 3. Hỏi người dùng muốn test bao nhiêu
    try:
        n_samples = int(input(f"👉 Bạn muốn test bao nhiêu mẫu? (Nhập số < {total_attacks}): "))
    except ValueError:
        print("Vui lòng nhập số nguyên hợp lệ!")
        return

    print(f"\n🚀 Đang lấy ngẫu nhiên {n_samples} mẫu để kiểm tra...")
    samples = attack_df.sample(n=n_samples, random_state=42) # random_state để kết quả cố định
    
    # 4. Chuẩn bị dữ liệu (Làm hàng loạt - Vectorization cho nhanh)
    print("⚙️  Đang tiền xử lý dữ liệu hàng loạt...")
    
    # Load model & scalers
    model = load_model(MODEL_PATH)
    scaler_time = joblib.load(SCALER_TIME_PATH)
    scaler_stat = joblib.load(SCALER_STAT_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    # Xử lý Input Time
    X_time = samples[TIME_FEATURES].values
    X_time = scaler_time.transform(X_time)
    X_time = X_time.reshape(X_time.shape[0], 1, len(TIME_FEATURES))

    # Xử lý Input Stat
    X_stat = samples[STAT_FEATURES].values
    X_stat = scaler_stat.transform(X_stat)

    # 5. Dự đoán (Batch Prediction)
    print("🧠 AI đang suy luận...")
    # Lưu ý: Thứ tự đúng là [X_time, X_stat]
    pred_probs = model.predict([X_time, X_stat], verbose=1)
    
    # Lấy nhãn dự đoán
    pred_indices = np.argmax(pred_probs, axis=1)
    pred_labels = le.inverse_transform(pred_indices)
    
    # Lấy nhãn thực tế
    true_labels = samples[LABEL_COLUMN].values

    # 6. Báo cáo kết quả
    accuracy = accuracy_score(true_labels, pred_labels)
    
    print("\n" + "="*50)
    print(f"📊 BÁO CÁO KẾT QUẢ KIỂM THỬ TRÊN {n_samples} MẪU")
    print("="*50)
    print(f"✅ Độ chính xác tổng thể: {accuracy * 100:.2f}%")
    print("-" * 50)
    
    # Đếm số lượng sai sót
    errors = samples[true_labels != pred_labels]
    print(f"❌ Số mẫu bị đoán sai: {len(errors)} / {n_samples}")
    
    if len(errors) > 0:
        print("\n🔍 CHI TIẾT CÁC CA SAI SÓT (Top 5):")
        # Tạo dataframe so sánh cho dễ nhìn
        comparison = pd.DataFrame({
            'Thực tế': true_labels,
            'AI đoán': pred_labels
        })
        # Lọc ra các dòng sai
        wrong_cases = comparison[comparison['Thực tế'] != comparison['AI đoán']]
        print(wrong_cases.head(5))
        
        print("\n📈 Thống kê chi tiết theo từng loại tấn công:")
        print(classification_report(true_labels, pred_labels, zero_division=0))

if __name__ == "__main__":
    evaluate_mass_attack()