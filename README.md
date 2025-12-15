<div align="center">

# 🛡️ Hệ thống Cảnh báo Sớm Tấn công Mạng Đa đầu vào

## Multi-Input Hybrid IDS (LSTM + DNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**Đồ án Mạng Máy Tính - HK251** _Giảng viên hướng dẫn: Thầy Bùi Xuân Giang_

</div>

---

# 📋 Phần 1: Mở đầu

## 📖 Giới thiệu

Dự án xây dựng một hệ thống phát hiện xâm nhập (IDS) lai ghép sử dụng kỹ thuật **Học sâu (Deep Learning)**. Hệ thống áp dụng chiến thuật **Feature Splitting** trên bộ dữ liệu chuẩn **CIC-IDS2017** để giả lập kiến trúc Đa đầu vào (Multi-Input):

- ⏱️ **Input A (Temporal):** Đặc trưng thời gian -> xử lý bởi **LSTM**.
- 📊 **Input B (Statistical):** Đặc trưng thống kê -> xử lý bởi **DNN**.

Mục tiêu: Phát hiện và phân loại chính xác các cuộc tấn công (DDoS, PortScan...) và đưa ra cảnh báo sớm.

---

## 🛠️ Cài đặt & Hướng dẫn chạy

### 1. Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **Bộ nhớ**: Khuyến nghị 8GB RAM trở lên (để xử lý dữ liệu CSV)

### 2. Cài đặt thư viện

Chạy lệnh sau để cài đặt các gói phụ thuộc:

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị Dữ liệu

- Tải bộ dữ liệu CIC-IDS2017.
- Đổi tên file thành CIC-IDS2017.csv.
- Di chuyển file vào thư mục: data/raw/.

## 🚀 Quy trình chạy (Workflow)

### Bước 1: Tiền xử lý dữ liệu (Preprocessing)

Script này sẽ đọc file CSV, làm sạch, tách đặc trưng thành 2 nhóm (Time & Stat) và lưu kết quả vào `data/processed/.`

```bash
python src/preprocess.py
```

### Bước 2: Huấn luyện Mô hình (Training)

Xây dựng mô hình Hybrid (LSTM + DNN), huấn luyện và lưu model vào `saved_models/`.

```bash
python src/train.py
```

### Bước 3: Chạy Hệ thống Cảnh báo (Alert System)

Load model đã train, giả lập luồng dữ liệu mới và in ra cảnh báo.

```bash
python src/alert_system.py
```

---

# 📋 Phần 2: Cấu trúc thư mục

---

## 📂 Cấu trúc Dự án

```text
Multi-Input_IDS/
│
├── data/
│   ├── raw/                  # Chứa file CIC-IDS2017.csv (sau khi gộp)
│   └── processed/            # Chứa file .npy sau khi tiền xử lý (để train nhanh)
│
├── saved_models/             # Nơi lưu model.h5 và các scaler (.pkl)
│
├── src/                      # Source code chính
│   ├── __init__.py           # Đánh dấu package
│   ├── utils.py              # Cấu hình chung (Tên cột, Đường dẫn)
│   ├── preprocess.py         # Code làm sạch, chuẩn hóa & tách đặc trưng
│   ├── model.py              # Kiến trúc mạng lai LSTM + DNN
│   ├── train.py              # Script huấn luyện mô hình
│   ├── demo_attack.py        # Demo tấn công giả lập (Visual Demo)
│   ├── evaluate_mass.py      # Script đánh giá diện rộng (Batch Testing)
│   └── alert_system.py       # Hệ thống cảnh báo & Dự đoán thời gian thực
│
├── requirements.txt          # Danh sách thư viện
└── README.md                 # Hướng dẫn sử dụng

```

---
