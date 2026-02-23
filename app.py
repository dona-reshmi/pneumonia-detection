import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #89f7fe, #66a6ff, #b0e0ff);
    color: #fff;
    margin: 0;
    padding: 0;
}

/* HERO */
.hero {
    text-align: center;
    padding: 100px 20px 60px 20px;
}
.hero-title {
    font-size: 70px;
    font-weight: 900;
    text-shadow: 2px 2px 15px rgba(0,0,0,0.2);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% {transform: scale(1);}
    50% {transform: scale(1.05);}
}
.hero-subtitle {
    font-size: 24px;
    color: #f0f9ff;
    margin-bottom: 40px;
}

/* SECTIONS */
.section {
    padding: 60px 20px;
    text-align: center;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 40px;
    margin: 20px auto;
    max-width: 450px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: transform 0.3s, box-shadow 0.3s;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

/* FEATURES GRID */
.features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 30px;
    justify-items: center;
    margin-top: 40px;
}
.feature-card {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 30px;
    width: 220px;
    box-shadow: 0 6px 25px rgba(0,0,0,0.2);
    transition: transform 0.3s, box-shadow 0.3s;
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 35px rgba(0,0,0,0.3);
}

/* RESULTS */
.result-text {
    font-size: 28px;
    font-weight: bold;
    margin-top: 15px;
}
.result-normal { color: #22c55e; }
.result-pneumonia { color: #ef4444; }

/* FOOTER */
.footer {
    text-align: center;
    padding: 30px 20px;
    margin-top: 60px;
    font-size: 14px;
    color: rgba(255,255,255,0.7);
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Pneumonia Detection in Seconds</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- UPLOAD & PREDICT ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("## 📤 Upload Your Chest X-Ray")

uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("Analyzing X-Ray..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if confidence > 0.5:
        st.markdown(f"<div class='result-text result-pneumonia'>🔴 Pneumonia Detected</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.markdown(f"<div class='result-text result-normal'>🟢 Normal Chest X-Ray</div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {(1-confidence)*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload an X-ray image to get diagnosis")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURES ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("## 🌟 Features")

st.markdown('<div class="features">', unsafe_allow_html=True)
features = [
    ("⚡ Fast Diagnosis", "Get results in seconds"),
    ("🎯 High Accuracy", "Trained on real chest X-ray data"),
    ("💻 Elegant UI", "Smooth and intuitive interface"),
    ("🧠 AI-Powered", "Deep learning based predictions"),
]
for icon, desc in features:
    st.markdown(f'<div class="feature-card"><h3>{icon}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("🧠 Developed with Deep Learning | 🏥 For Educational Use Only")
st.markdown('</div>', unsafe_allow_html=True)
