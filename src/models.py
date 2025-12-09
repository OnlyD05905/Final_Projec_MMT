import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate

def build_hybrid_model(time_shape, stat_shape, num_classes):
    print("--- [GĐ2] Đang xây dựng Model Đa đầu vào... ---")
    
    # === NHÁNH A: Temporal (LSTM) ===
    input_A = Input(shape=time_shape, name="Input_Time")
    x_a = LSTM(64, return_sequences=False)(input_A)
    x_a = Dense(32, activation='relu')(x_a)

    # === NHÁNH B: Statistical (DNN) ===
    input_B = Input(shape=stat_shape, name="Input_Stat")
    x_b = Dense(128, activation='relu')(input_B)
    x_b = Dropout(0.2)(x_b)
    x_b = Dense(64, activation='relu')(x_b)

    # === HỢP NHẤT (Concatenate) ===
    combined = Concatenate()([x_a, x_b])

    # === CLASSIFIER HEAD ===
    z = Dense(64, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax', name="Output_Risk")(z)

    # Compile
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model