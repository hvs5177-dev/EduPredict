import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="EduPredict AI", layout="wide")

# --------------------------
# LOGIN SYSTEM
# --------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("## 🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# --------------------------
# LOAD MODEL
# --------------------------
model = load_model("models/student_model.h5", compile=False)
scaler = joblib.load("models/scaler.pkl")

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------
st.sidebar.title("🎓 EduPredict AI")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📊 Analytics", "📄 Report", "🚪 Logout"]
)

if menu == "🚪 Logout":
    st.session_state.logged_in = False
    st.rerun()

# --------------------------
# DASHBOARD
# --------------------------
if menu == "🏠 Dashboard":

    st.title("🎯 Student Performance Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 25, 18)
        study_hours = st.number_input("Study Hours", 0.0, 12.0)
        attendance = st.number_input("Attendance (%)", 0.0, 100.0)

    with col2:
        previous = st.number_input("Previous Score", 0.0, 100.0)

    math = st.slider("Math", 0, 100, 50)
    science = st.slider("Science", 0, 100, 50)
    english = st.slider("English", 0, 100, 50)

    # --------------------------
    # VALIDATION
    # --------------------------
    if previous == 0 and (math > 0 or science > 0 or english > 0):
        st.warning("⚠ Invalid input: Previous score cannot be 0 when subjects are non-zero")

    if st.button("🚀 Predict"):

        if previous == 0 and (math > 0 or science > 0 or english > 0):
            st.error("Fix input first")
            st.stop()

        input_data = np.array([[age, study_hours, attendance, math, science, english, previous]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        predicted_marks = float(prediction[0][0])
        predicted_marks = max(0, min(100, predicted_marks))

        lower = max(0, round(predicted_marks - 4))
        upper = min(100, round(predicted_marks + 4))

        if math < 20 or science < 20 or english < 20:
            result = "FAIL"
            risk = "High Risk"
        else:
            if predicted_marks >= 75:
                risk = "Low Risk"
            elif predicted_marks >= 50:
                risk = "Medium Risk"
            else:
                risk = "High Risk"

            result = "PASS" if predicted_marks >= 40 else "FAIL"

        # STORE SESSION
        st.session_state.prediction = {
            "marks": predicted_marks,
            "range": f"{lower}-{upper}",
            "risk": risk,
            "result": result,
            "subjects": [math, science, english]
        }

    # --------------------------
    # RESULT DISPLAY
    # --------------------------
    if "prediction" in st.session_state:

        data = st.session_state.prediction

        st.markdown("### 📊 Result Dashboard")

        c1, c2, c3 = st.columns(3)

        c1.metric("🎯 Marks", round(data["marks"], 2))
        c2.metric("📈 Range", data["range"])
        c3.metric("⚠ Risk", data["risk"])

        if data["result"] == "PASS":
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

        st.progress(int(data["marks"]))

# --------------------------
# ANALYTICS
# --------------------------
elif menu == "📊 Analytics":

    st.title("📊 Performance Analytics")

    if "prediction" not in st.session_state:
        st.info("Run prediction first")
    else:
        data = st.session_state.prediction

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            labels = ["Math", "Science", "English"]
            ax.bar(labels, data["subjects"])
            ax.set_ylim(0, 100)
            ax.set_title("Subject Scores")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            avg = sum(data["subjects"]) / 3
            ax2.plot(["Average", "Predicted"], [avg, data["marks"]], marker='o')
            ax2.set_ylim(0, 100)
            ax2.set_title("Performance Comparison")
            st.pyplot(fig2)

# --------------------------
# REPORT
# --------------------------
elif menu == "📄 Report":

    st.title("📄 Report Generation")

    if "prediction" not in st.session_state:
        st.info("Run prediction first")
    else:
        d = st.session_state.prediction

        report = f"""
        STUDENT PERFORMANCE REPORT

        Marks: {round(d['marks'],2)}
        Range: {d['range']}
        Risk: {d['risk']}
        Result: {d['result']}
        """

        st.download_button(
            "📥 Download Report",
            data=report,
            file_name="report.txt"
        )