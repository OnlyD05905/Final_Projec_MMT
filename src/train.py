from src.preprocess import load_and_preprocess
from src.model import build_hybrid_model
from src.utils import MODEL_PATH

def main_train():
    # 1. Lấy dữ liệu từ Member 1
    X_time_train, X_stat_train, X_time_test, X_stat_test, y_train, y_test, num_classes = load_and_preprocess()

    # Lấy shape động (để không phải hardcode)
    time_shape = (X_time_train.shape[1], X_time_train.shape[2]) # (1, n_features)
    stat_shape = (X_stat_train.shape[1],)                       # (n_features,)

    # 2. Xây dựng model
    model = build_hybrid_model(time_shape, stat_shape, num_classes)
    model.summary()

    # 3. Huấn luyện (Truyền list 2 input)
    print("--- [GĐ3] Bắt đầu Training... ---")
    history = model.fit(
        [X_time_train, X_stat_train], 
        y_train,
        validation_data=([X_time_test, X_stat_test], y_test),
        epochs=50, 
        batch_size=64
    )

    # 4. Lưu model
    model.save(MODEL_PATH)
    print(f"--- Đã lưu model tại: {MODEL_PATH} ---")

if __name__ == "__main__":
    main_train()