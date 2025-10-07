import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback

# ==============================
# ⚙️ PAGE CONFIG
# ==============================
st.set_page_config(page_title="Vasopressor Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Vasopressor Prediction App")
st.write("Ứng dụng dự đoán khả năng cần sử dụng **vasopressor** cho bệnh nhân ICU.")

# ==============================
# 📁 LOAD MODEL FUNCTION
# ==============================
def load_model(model_name):
    path = f"{model_name}_vasopressor_model.pkl"
    if not os.path.exists(path):
        st.error(f"❌ Không tìm thấy mô hình: {path}")
        return None
    try:
        model = joblib.load(path)
        st.sidebar.success(f"✅ Đã load mô hình: {model_name}")
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi load mô hình {model_name}: {e}")
        st.text(traceback.format_exc())
        return None

# ==============================
# 📁 LOAD SCALER (nếu có)
# ==============================
scaler = None
if os.path.exists("vasopressor_scaler.pkl"):
    try:
        scaler = joblib.load("vasopressor_scaler.pkl")
        st.sidebar.info("⚙️ Scaler được load thành công.")
    except Exception as e:
        st.sidebar.warning(f"Không thể load scaler: {e}")

# ==============================
# 🧠 MODEL SELECTION
# ==============================
st.sidebar.header("🧩 Chọn mô hình")
model_choice = st.sidebar.selectbox(
    "Chọn mô hình dự đoán:",
    ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]
)
model = load_model(model_choice)

# ==============================
# 📊 INPUT FORM
# ==============================
st.header("📋 Nhập dữ liệu bệnh nhân")

age = st.number_input("Tuổi", 0, 120, 65)
heartrate = st.number_input("Nhịp tim (bpm)", 0, 250, 90)
systemicmean = st.number_input("Mean Arterial Pressure (MAP)", 0, 200, 70)
sao2 = st.number_input("SaO₂ (%)", 0, 100, 95)
lactate = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 2.0)

# ==============================
# 🧮 CREATE INPUT DATAFRAME
# ==============================
user_input = pd.DataFrame({
    "age": [age],
    "heartrate": [heartrate],
    "systemicmean": [systemicmean],
    "sao2": [sao2],
    "lactate": [lactate]
})

if scaler is not None:
    try:
        X_input = pd.DataFrame(scaler.transform(user_input), columns=user_input.columns)
    except Exception as e:
        st.warning(f"Scaler không tương thích: {e}")
        X_input = user_input.copy()
else:
    X_input = user_input.copy()

# ==============================
# 📈 LOAD MODEL PERFORMANCE
# ==============================
metrics_file = "vasopressor_model_comparison.csv"
if os.path.exists(metrics_file):
    results_df = pd.read_csv(metrics_file)
    # Xử lý các cột mới (có thể tên khác: AUC, Brier thay vì AUC_mean,...)
    auc_col = "AUC_mean" if "AUC_mean" in results_df.columns else "AUC"
    brier_col = "Brier_mean" if "Brier_mean" in results_df.columns else "Brier"
    row = results_df[results_df["Model"].str.contains(model_choice, case=False, na=False)]
    if not row.empty:
        auc = row[auc_col].values[0]
        brier = row[brier_col].values[0]
        st.sidebar.markdown(f"**📈 AUC:** {auc:.3f}")
        st.sidebar.markdown(f"**🎯 Brier Score:** {brier:.3f}")
    else:
        st.sidebar.info("📊 Không có dữ liệu hiệu năng cho mô hình này.")
else:
    st.sidebar.info("📊 Không tìm thấy file `vasopressor_model_comparison.csv`.")

# ==============================
# 🔮 PREDICT BUTTON
# ==============================
if st.button("🧠 Dự đoán"):
    if model is None:
        st.error("⚠️ Mô hình chưa được load thành công. Vui lòng kiểm tra lại file.")
    else:
        try:
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else None

            st.success(f"✅ Kết quả: {'Cần vasopressor' if pred == 1 else 'Không cần vasopressor'}")

            if prob is not None:
                st.write(f"🔹 Xác suất cần vasopressor: **{prob:.2%}**")
                st.progress(float(prob))

                # --- Biểu đồ xác suất ---
                st.subheader("📊 Xác suất dự đoán")
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.barh(["Vasopressor"], [prob], color="red" if prob > 0.5 else "green")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Xác suất")
                ax.set_title("Khả năng cần dùng vasopressor")
                for i, v in enumerate([prob]):
                    ax.text(v + 0.02, i, f"{v:.2%}", va="center", fontweight="bold")
                st.pyplot(fig)

            # --- So sánh giá trị ---
            st.subheader("📈 So sánh thông số bệnh nhân")
            normal_ranges = {
                "age": (20, 80),
                "heartrate": (60, 100),
                "systemicmean": (70, 105),
                "sao2": (94, 100),
                "lactate": (0.5, 2.2)
            }

            values = [age, heartrate, systemicmean, sao2, lactate]
            features = ["Tuổi", "Nhịp tim", "MAP", "SaO₂", "Lactate"]
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
            st.text(traceback.format_exc())
