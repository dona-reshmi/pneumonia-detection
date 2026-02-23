import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- INNOVATIVE CSS ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #89f7fe, #66a6ff);
    color: #fff;
}

.hero {
    text-align: center;
    padding: 100px 20px 60px 20px;
    animation: fadeIn 2s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}

.hero-title {
    font-size: 70px;
    font-weight: 900;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% {transform: scale(1);}
    50% {transform: scale(1.05);}
}

.hero-subtitle {
    font-size: 24px;
    color: #f0f9ff;
    margin-bottom: 40px;
}

.section {
    padding: 60px 40px;
    text-align: center;
}

.card {
    background: rgba(255,255,255,0.08);
    border-radius: 25px;
    padding: 40px;
    margin: 20px auto;
    box-shadow: 0 0 30px rgba(0,0,0,0.2);
    max-width: 500px;
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-10px);
    box-shadow: 0 0 40px rgba(0,0,0,0.4);
}

.result-normal {
    color: #22c55e;
    font-size: 28px;
    font-weight: bold;
}

.result-pneumonia {
    color: #ef4444;
    font-size: 28px;
    font-weight: bold;
}

.features {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 25px;
}

.feature-card {
    background: rgba(255,255,255,0.08);
    border-radius: 25px;
    padding: 30px;
    width: 250px;
    box-shadow: 0 0 20px rgba(0,0,0,0.2);
    transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 0 35px rgba(0,0,0,0.35);
}

.circular-progress {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 10px solid rgba(255,255,255,0.2);
    border-top: 10px solid #fff;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% {transform: rotate(0deg);}
    100% {transform: rotate(360deg);}
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
        st.markdown(f"<div class='result-pneumonia'>🔴 Pneumonia Detected</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='circular-progress'></div>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.markdown(f"<div class='result-normal'>🟢 Normal Chest X-Ray</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='circular-progress'></div>", unsafe_allow_html=True)
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
    ("⚡ Fast Diagnosis", "Results in seconds"),
    ("🎯 High Accuracy", "Trained on chest X-ray datasets"),
    ("💻 Elegant UI", "Smooth and intuitive interface"),
    ("🧠 AI-Powered", "Deep learning based predictions")
]

for icon, desc in features:
    st.markdown(f'<div class="feature-card"><h3>{icon}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("<hr>")
st.markdown("<center>🧠 Developed with Deep Learning | 🏥 For Educational Use Only</center>")
st.markdown('</div>', unsafe_allow_html=True)
