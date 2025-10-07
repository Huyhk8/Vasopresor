import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Cấu hình trang ---
st.set_page_config(page_title="Vasopressor Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Vasopressor Prediction App")
st.write("Ứng dụng dự đoán khả năng cần sử dụng **vasopressor** cho bệnh nhân ICU.")

# --- Load mô hình ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("vasopressor_predictor.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Không thể load mô hình: {e}")
        st.stop()

model = load_model()

# --- Lấy danh sách feature của mô hình ---
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = ["age", "heartrate", "map", "spo2", "lactate"]

# --- Nhập liệu người dùng ---
st.header("📋 Nhập dữ liệu bệnh nhân")

age = st.number_input("Tuổi", min_value=0, max_value=120, value=65)
heartrate = st.number_input("Nhịp tim (bpm)", min_value=0, max_value=250, value=90)
map_value = st.number_input("Mean Arterial Pressure (MAP)", min_value=0, max_value=200, value=70)
spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, value=95)
lactate = st.number_input("Lactate (mmol/L)", min_value=0.0, max_value=20.0, value=2.0)

# --- Chuẩn bị input cho mô hình ---
user_input = pd.DataFrame({
    "age": [age],
    "heartrate": [heartrate],
    "map": [map_value],
    "spo2": [spo2],
    "lactate": [lactate]
})

# Chuẩn hóa để khớp tên cột mô hình
X_input = pd.DataFrame(columns=model_features)
for col in model_features:
    X_input[col] = user_input[col] if col in user_input.columns else [0]

# --- Nút dự đoán ---
if st.button("🧠 Dự đoán"):
    try:
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

        st.success(f"✅ Kết quả: {'Cần vasopressor' if pred == 1 else 'Không cần vasopressor'}")

        if prob is not None:
            st.write(f"🔹 Xác suất cần vasopressor: **{prob:.2%}**")
            st.progress(float(prob))

            # --- Vẽ biểu đồ thanh thể hiện xác suất ---
            st.subheader("📊 Xác suất dự đoán")
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Vasopressor"], [prob], color="red" if prob > 0.5 else "green")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Xác suất")
            ax.set_title("Khả năng cần dùng vasopressor")
            for i, v in enumerate([prob]):
                ax.text(v + 0.02, i, f"{v:.2%}", va="center", fontweight="bold")
            st.pyplot(fig)

        # --- Biểu đồ so sánh giá trị bệnh nhân ---
        st.subheader("📈 So sánh thông số bệnh nhân")
        normal_ranges = {
            "age": (20, 80),
            "heartrate": (60, 100),
            "map": (70, 105),
            "spo2": (94, 100),
            "lactate": (0.5, 2.2)
        }

        values = [age, heartrate, map_value, spo2, lactate]
        features = ["Tuổi", "Nhịp tim", "MAP", "SpO₂", "Lactate"]
        normal_low = [normal_ranges[f][0] for f in normal_ranges]
        normal_high = [normal_ranges[f][1] for f in normal_ranges]

        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.bar(features, values, color="skyblue", label="Giá trị bệnh nhân")
        ax2.plot(features, normal_low, "g--", label="Giới hạn thấp (bình thường)")
        ax2.plot(features, normal_high, "r--", label="Giới hạn cao (bình thường)")
        ax2.set_ylabel("Giá trị")
        ax2.set_title("So sánh thông số với khoảng bình thường")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"⚠️ Lỗi khi dự đoán: {e}")
