import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🧠", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.main {
    background: transparent;
}

.hero {
    text-align: center;
    padding-top: 40px;
    padding-bottom: 30px;
}

.hero h1 {
    font-size: 60px;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5ff, #00ff87);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 20px;
    color: #d1d5db;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 40px;
    margin-top: 40px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

.result-box {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown("""
<div class="hero">
    <h1>🧠 PneumoAI</h1>
    <p>Next-Generation AI System for Intelligent Pneumonia Detection</p>
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

# ---------- GLASS CARD ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("AI analyzing medical image..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if confidence > 0.5:
        st.markdown(f"🔴 **Pneumonia Detected**  \nConfidence: {confidence*100:.2f}%")
    else:
        st.markdown(f"🟢 **Normal Chest X-Ray**  \nConfidence: {(1-confidence)*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
