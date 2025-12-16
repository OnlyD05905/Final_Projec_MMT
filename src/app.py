# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MODEL_PATH, SCALER_TIME_PATH, SCALER_STAT_PATH, LABEL_ENCODER_PATH
from utils import TIME_FEATURES, STAT_FEATURES, LABEL_COLUMN

# --- CẤU HÌNH ---
st.set_page_config(page_title="IDS Dashboard", layout="wide")
CHUNK_SIZE = 10000  
SAMPLE_SIZE = 100  

# Load các tài nguyên (Chỉ load 1 lần)
@st.cache_resource
def load_resources():
    try:
        # Load hybrid model (2 inputs)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Load scalers cho cả time và stat features
        scaler_time = joblib.load(SCALER_TIME_PATH)
        scaler_stat = joblib.load(SCALER_STAT_PATH)
        
        # Load label encoder
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        return model, scaler_time, scaler_stat, label_encoder
    except Exception as e:
        st.error(f"Không thể tải file hệ thống. Bạn đã chạy train chưa? Lỗi: {e}")
        st.error(f"Chi tiết lỗi: {e}")
        return None

resources = load_resources()

if resources:
    model, scaler_time, scaler_stat, label_encoder = resources
    
    # Tự động tìm nhãn BENIGN từ label encoder
    BENIGN_LABEL_NAME = "BENIGN"
    label_classes = label_encoder.classes_
    for label in label_classes:
        if "BENIGN" in label.upper():
            BENIGN_LABEL_NAME = label
            break
    
    # Khởi tạo bộ đếm chi tiết cho từng loại nhãn
    stats = {label: 0 for label in label_classes} 
    
    st.title("Intrusion Detection System (IDS) - Hybrid Model")
    
    st.markdown("""
    **Expected CSV Format:** The uploaded CSV should contain network traffic features including:
    - Time features: Flow Duration, Flow IAT Mean, Flow IAT Std, etc.
    - Statistical features: Total Fwd Packets, Total Backward Packets, etc.
    - Label column: Contains attack types (BENIGN, DDoS, PortScan, etc.)
    
    **Note:** The model was trained on CIC-IDS2017 dataset format.
    """)
    
    # --- SIMULATION SECTION ---
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file for testing (Max 1GB)", type="csv")
    
    # Configuration options
    sample_size = st.sidebar.slider("Sample Size", min_value=10, max_value=500, value=100, step=10)
    processing_speed = st.sidebar.selectbox("Processing Speed", 
                                          options=["Fast", "Normal", "Slow"], 
                                          index=1,
                                          help="Fast: 0.05s delay, Normal: 0.1s delay, Slow: 0.2s delay")
    
    # Map speed to delay
    speed_delays = {"Fast": 0.05, "Normal": 0.1, "Slow": 0.2}
    delay = speed_delays[processing_speed]
    
    # Update global sample size
    SAMPLE_SIZE = sample_size
    
    if uploaded_file is not None:
        st.info(f"Processing file '{uploaded_file.name}'. Reading {SAMPLE_SIZE} random samples...")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Real-time Monitoring", "Detailed Results", "Statistics"])
        
        logs = []
        results_data = []
        
        try:
            # Use same encoding as preprocess.py and handle potential issues
            df_reader = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE, encoding='cp1252', 
                                   on_bad_lines='skip', low_memory=False)
            first_chunk = next(df_reader)
            
            # Clean column names (remove whitespace)
            first_chunk.columns = first_chunk.columns.str.strip()
            
            # Check for required columns
            required_columns = TIME_FEATURES + STAT_FEATURES + [LABEL_COLUMN]
            missing_columns = [col for col in required_columns if col not in first_chunk.columns]
            
            if missing_columns:
                st.error(f"Missing required columns in CSV: {missing_columns}")
                st.error("Please ensure your CSV file contains all the required features.")
                st.stop()
            
            if len(first_chunk) < SAMPLE_SIZE:
                df_sample = first_chunk
            else:
                df_sample = first_chunk.sample(SAMPLE_SIZE)
                
                st.success(f"Loaded {len(df_sample)} random samples from the first {CHUNK_SIZE} rows for analysis.")

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Use a separate counter for progress bar
            processed_count = 0
            
            for idx, row in df_sample.iterrows():
                processed_count += 1
                status_text.text(f"Processing sample {processed_count}/{len(df_sample)}")
                progress_bar.progress(processed_count / len(df_sample))
                
                row_df = pd.DataFrame([row], columns=df_sample.columns)
                
                # Clean column names (remove whitespace)
                row_df.columns = row_df.columns.str.strip()
                row_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                row_df.fillna(0, inplace=True)

                try:
                    # Extract time and stat features (only use available columns)
                    available_time_features = [col for col in TIME_FEATURES if col in row_df.columns]
                    available_stat_features = [col for col in STAT_FEATURES if col in row_df.columns]
                    
                    if not available_time_features or not available_stat_features:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] | ERROR: Missing required features")
                        continue
                    
                    X_time_raw = row_df[available_time_features].values
                    X_stat_raw = row_df[available_stat_features].values
                    
                    # Scale using loaded scalers (handle missing features)
                    try:
                        X_time_scaled = scaler_time.transform(X_time_raw)
                        X_stat_scaled = scaler_stat.transform(X_stat_raw)
                    except ValueError as e:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] | ERROR: Feature mismatch - {str(e)}")
                        continue
                    
                    # Reshape time features for LSTM (samples, 1, features)
                    X_time_scaled = X_time_scaled.reshape(1, 1, X_time_scaled.shape[1])
                    # Reshape stat features for DNN (samples, features)
                    X_stat_scaled = X_stat_scaled.reshape(1, -1)
                    
                except Exception as e:
                    logs.insert(0, f"[{time.strftime('%H:%M:%S')}] | PROCESSING ERROR ROW {idx}: {e}")
                    continue

                # Make prediction using hybrid model (2 inputs)
                try:
                    pred_probs = model.predict([X_time_scaled, X_stat_scaled], verbose=0)
                    pred_id = np.argmax(pred_probs)
                    pred_label = label_encoder.classes_[pred_id]
                    confidence = float(pred_probs[0, pred_id]) * 100
                    
                    # Get confidence for all classes
                    all_confidences = {label_encoder.classes_[i]: float(pred_probs[0, i]) * 100 
                                     for i in range(len(label_encoder.classes_))}
                    
                except Exception as e:
                    logs.insert(0, f"[{time.strftime('%H:%M:%S')}] | PREDICTION ERROR: {e}")
                    continue
                
                status = pred_label
                color = "green" if pred_label == BENIGN_LABEL_NAME else "red"
                
                if pred_label != BENIGN_LABEL_NAME:
                    status = f"ALERT: {pred_label}"
                
                # Update statistics
                stats[pred_label] += 1
                
                # Get true label
                true_label = 'N/A'
                for col in df_sample.columns:
                    if col.strip().lower() == 'label':
                        true_label = row.get(col, 'N/A')
                        if isinstance(true_label, str): 
                            true_label = true_label.strip()
                        break

                # Store detailed results
                result_row = {
                    'Timestamp': time.strftime('%H:%M:%S'),
                    'Row_Index': idx,
                    'Prediction': pred_label,
                    'Confidence': f"{confidence:.1f}%",
                    'True_Label': true_label,
                    'Status': 'Normal' if pred_label == BENIGN_LABEL_NAME else 'Attack',
                    'All_Confidences': all_confidences
                }
                
                # Add key feature values (only for available features)
                for i, feature in enumerate(available_time_features):
                    result_row[f'Time_{feature}'] = float(X_time_raw[0, i])
                for i, feature in enumerate(available_stat_features):
                    result_row[f'Stat_{feature}'] = float(X_stat_raw[0, i])
                    
                results_data.append(result_row)

                msg = f"[{time.strftime('%H:%M:%S')}] | Confidence: {confidence:.1f}% | Prediction: {status} | Actual: {true_label}"
                logs.insert(0, msg)
                
                time.sleep(delay)  # Use configurable delay

            progress_bar.empty()
            status_text.empty()
            
            # Display results in tabs
            with tab1:
                st.subheader("Real-time Monitoring Log")
                log_placeholder = st.empty()
                with log_placeholder.container():
                    st.code('\n'.join(logs[:20]), language=None)
            
            with tab2:
                st.subheader("Detailed Prediction Results")
                if results_data:
                    # Convert to DataFrame for display
                    results_df = pd.DataFrame(results_data)
                    
                    # Show summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_samples = len(results_df)
                        st.metric("Total Samples", total_samples)
                    with col2:
                        attack_count = len(results_df[results_df['Status'] == 'Attack'])
                        st.metric("Detected Attacks", attack_count)
                    with col3:
                        accuracy = (len(results_df[results_df['Prediction'] == results_df['True_Label']]) / total_samples * 100) if total_samples > 0 else 0
                        st.metric("Accuracy", f"{accuracy:.1f}%")
                    
                    # Show detailed table
                    st.dataframe(results_df[['Timestamp', 'Row_Index', 'Prediction', 'Confidence', 'True_Label', 'Status']], 
                               use_container_width=True)
                    
                    # Show confidence distribution for a selected sample
                    if len(results_data) > 0:
                        sample_idx = st.selectbox("Select sample to view confidence details:", 
                                                range(len(results_data)), 
                                                format_func=lambda x: f"Sample {x} - {results_data[x]['Prediction']}")
                        
                        if sample_idx is not None:
                            selected_result = results_data[sample_idx]
                            st.subheader(f"Confidence Scores for Sample {sample_idx}")
                            
                            # Create confidence chart
                            conf_df = pd.DataFrame({
                                'Class': list(selected_result['All_Confidences'].keys()),
                                'Confidence': list(selected_result['All_Confidences'].values())
                            })
                            conf_df = conf_df.sort_values('Confidence', ascending=False)
                            st.bar_chart(conf_df.set_index('Class'))
                            
                            # Show feature values
                            st.subheader("Feature Values Used")
                            time_features = {k.replace('Time_', ''): v for k, v in selected_result.items() if k.startswith('Time_')}
                            stat_features = {k.replace('Stat_', ''): v for k, v in selected_result.items() if k.startswith('Stat_')}
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Time Features:**")
                                st.json(time_features)
                            with col2:
                                st.write("**Statistical Features:**")
                                st.json(stat_features)
            
            with tab3:
                st.subheader("Detection Statistics")
                
                # Overall statistics
                total_processed = sum(stats.values())
                benign_count = stats.get(BENIGN_LABEL_NAME, 0)
                attack_types = {k: v for k, v in stats.items() if k != BENIGN_LABEL_NAME and v > 0}
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", total_processed)
                with col2:
                    st.metric("Normal Traffic", benign_count)
                with col3:
                    st.metric("Attack Traffic", sum(attack_types.values()))
                with col4:
                    attack_rate = (sum(attack_types.values()) / total_processed * 100) if total_processed > 0 else 0
                    st.metric("Attack Rate", f"{attack_rate:.1f}%")
                
                # Attack type breakdown
                if attack_types:
                    st.subheader("Attack Type Distribution")
                    attack_df = pd.DataFrame({
                        'Attack Type': list(attack_types.keys()),
                        'Count': list(attack_types.values()),
                        'Percentage': [f"{v/total_processed*100:.1f}%" for v in attack_types.values()]
                    })
                    st.dataframe(attack_df, use_container_width=True)
                    
                    # Pie chart for attack types
                    st.bar_chart(attack_df.set_index('Attack Type')['Count'])
                else:
                    st.info("No attacks detected in the analyzed samples.")

        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")