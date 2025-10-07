import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load mô hình ---
@st.cache_resource
def load_model():
    return joblib.load("vasopressor_predictor.pkl")

model = load_model()

st.title("🧬 Vasopressor Prediction App")
st.write("Ứng dụng dự đoán khả năng cần sử dụng **vasopressor** cho bệnh nhân ICU.")

# --- Nhập dữ liệu ---
st.header("📋 Nhập dữ liệu bệnh nhân")

age = st.number_input("Tuổi", min_value=0, max_value=120, value=65)
heartrate = st.number_input("Nhịp tim (bpm)", min_value=0, max_value=250, value=90)
map_value = st.number_input("Mean Arterial Pressure (MAP)", min_value=0, max_value=200, value=70)
spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, value=95)
lactate = st.number_input("Lactate (mmol/L)", min_value=0.0, max_value=20.0, value=2.0)

input_data = pd.DataFrame({
    "age": [age],
    "heartrate": [heartrate],
    "map": [map_value],
    "spo2": [spo2],
    "lactate": [lactate]
})

# --- Dự đoán ---
if st.button("🧠 Dự đoán"):
    try:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        st.success(f"✅ Kết quả: {'Cần vasopressor' if pred == 1 else 'Không cần vasopressor'}")
        st.write(f"🔹 Xác suất: {prob:.2%}")
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
