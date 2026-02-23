import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- REMOVE STREAMLIT DEFAULT HEADER ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- GLOBAL STYLING ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2f3, #d9e4f5);
}

.hero {
    text-align: center;
    padding-top: 80px;
    padding-bottom: 40px;
}

.hero h1 {
    font-size: 65px;
    font-weight: 800;
    color: #1f2937;
}

.hero p {
    font-size: 20px;
    color: #4b5563;
}

.upload-section {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    width: 60%;
    margin: auto;
}

.result-box {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown("""
<div class="hero">
    <h1>🫁 PneumoAI</h1>
    <p>AI-Powered Pneumonia Detection System</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- UPLOAD CARD ----------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Resize display smaller
    st.image(image, width=300)

    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("Analyzing medical image..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if confidence > 0.5:
        st.markdown(f"🔴 Pneumonia Detected<br>Confidence: {confidence*100:.2f}%", unsafe_allow_html=True)
    else:
        st.markdown(f"🟢 Normal Chest X-Ray<br>Confidence: {(1-confidence)*100:.2f}%", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
