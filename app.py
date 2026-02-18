import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tanzania Fish Price Dashboard",
    page_icon="🇹🇿",
    layout="wide"
)

# ---------------- CUSTOM CSS (Tanzania Theme) ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f9f4;
}

h1, h2, h3 {
    color: #1b5e20;
}

.sidebar .sidebar-content {
    background-color: #000000;
}

div.stButton > button {
    background-color: #fdd835;
    color: black;
    font-weight: bold;
    border-radius: 8px;
}

.metric-box {
    background-color: #1b5e20;
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- HEADER ----------------
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Flag_of_Tanzania.svg/320px-Flag_of_Tanzania.svg.png", width=150)

with col2:
    st.title("Tanzania Fish Price Prediction Dashboard")
    st.write("Advanced Machine Learning Dashboard for Fish Market Price Estimation")

st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Fish Input Parameters")

weight = st.sidebar.number_input("Fish Weight (kg)", min_value=0.0)
length = st.sidebar.number_input("Fish Length (cm)", min_value=0.0)
height = st.sidebar.number_input("Fish Height (cm)", min_value=0.0)
width = st.sidebar.number_input("Fish Width (cm)", min_value=0.0)

predict_btn = st.sidebar.button("Predict Price")

# ---------------- MAIN DASHBOARD ----------------
if predict_btn:

    features = np.array([[weight, length, height, width]])
    prediction = model.predict(features)[0]

    # KPI Metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Predicted Price (Tsh)", f"{prediction:,.0f}")

    with m2:
        st.metric("Weight (kg)", weight)

    with m3:
        st.metric("Length (cm)", length)

    st.markdown("---")

    # Visualization
    st.subheader("Price Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Predicted Price"], [prediction])
    ax.set_ylabel("Price (Tsh)")
    st.pyplot(fig)

    # Prediction Card
    st.markdown(f"""
    <div class="metric-box">
        <h2>Estimated Market Value</h2>
        <h1>Tsh {prediction:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed by Group No.32 | 2026 | Machine Learning Production App")
