import pandas as pd
import glob
import os

def merge_csv_files():
    # Đường dẫn đến thư mục chứa 8 file CSV con
    # (Bạn kiểm tra lại tên thư mục giải nén xem là MachineLearningCSV hay MachineLearningCVE nhé)
    input_path = "data/raw/MachineLearningCVE" 
    output_file = "data/raw/CIC-IDS2017.csv"

    print(f"🔍 Đang tìm file CSV trong: {input_path}")
    all_files = glob.glob(os.path.join(input_path, "*.csv"))

    if not all_files:
        print("❌ LỖI: Không tìm thấy file CSV nào! Hãy kiểm tra lại đường dẫn.")
        return

    print(f"✅ Tìm thấy {len(all_files)} file. Đang tiến hành gộp...")
    
    df_list = []
    for filename in all_files:
        print(f"  -> Đang đọc: {os.path.basename(filename)}")
        try:
            # Đọc file, bỏ qua các dòng lỗi mã hóa (nếu có)
            df = pd.read_csv(filename, index_col=None, header=0, encoding='cp1252')
            df_list.append(df)
        except Exception as e:
            print(f"  ⚠️ Lỗi khi đọc file {filename}: {e}")

    # Gộp lại
    print("⏳ Đang ghép nối dữ liệu (việc này tốn khoảng 1-2 phút)...")
    frame = pd.concat(df_list, axis=0, ignore_index=True)
    
    # Lưu ra file đích
    frame.to_csv(output_file, index=False)
    print(f"🎉 THÀNH CÔNG! File tổng đã được tạo tại: {output_file}")
    print(f"📊 Tổng số dòng dữ liệu: {len(frame)}")

if __name__ == "__main__":
    merge_csv_files()