import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

st.set_page_config(page_title="PneumoAI", layout="wide")

# Hide Streamlit default elements
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}

.section {
    padding: 80px 10%;
}

.hero-title {
    font-size: 60px;
    font-weight: 800;
}

.hero-sub {
    font-size: 20px;
    color: #d1d5db;
}

.feature-card {
    background: rgba(255,255,255,0.05);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    text-align: center;
    transition: 0.3s;
}

.feature-card:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.1);
}

.button-style {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 12px 28px;
    border-radius: 30px;
    font-size: 18px;
    text-decoration: none;
}

.upload-section {
    margin-top: 40px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="section">
    <div class="hero-title">🫁 PneumoAI</div>
    <div class="hero-sub">
        AI-powered Pneumonia Detection from Chest X-rays<br>
        Fast • Accurate • Intelligent
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURES ----------------
st.markdown('<div class="section">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="feature-card">⚡<br><b>Fast Diagnosis</b><br>Instant results in seconds</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">🎯<br><b>Accurate Model</b><br>Trained on thousands of X-rays</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-card">💻<br><b>Elegant Interface</b><br>Modern & clean design</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="feature-card">🧠<br><b>AI Technology</b><br>Deep learning powered</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ABOUT ----------------
st.markdown("""
<div class="section" style="text-align:center;">
    <h2>About PneumoAI</h2>
    <p style="max-width:800px; margin:auto; color:#cbd5e1;">
    PneumoAI is an advanced deep learning system designed to detect pneumonia
    from chest X-ray images with high precision. The model analyzes medical
    imaging patterns and provides instant diagnostic support.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="section upload-section">', unsafe_allow_html=True)
st.markdown("<h2>Get Started</h2>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    img = image.resize((150,150))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("Analyzing with AI..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(f"Pneumonia Detected (Confidence: {confidence*100:.2f}%)")
    else:
        st.success(f"Normal (Confidence: {(1-confidence)*100:.2f}%)")

st.markdown('</div>', unsafe_allow_html=True)
