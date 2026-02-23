import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- REMOVE STREAMLIT DEFAULT STYLING ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
}

.title {
    text-align: center;
    font-size: 70px;
    font-weight: 800;
    margin-top: 80px;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    margin-bottom: 60px;
    color: #dbeafe;
}

.center-content {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.result-text {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Pneumonia Detection System</div>', unsafe_allow_html=True)

st.markdown('<div class="center-content">', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # smaller display image
    st.image(image, width=280)

    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("Analyzing..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.markdown(
            f'<div class="result-text">🔴 Pneumonia Detected<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-text">🟢 Normal Chest X-Ray<br>Confidence: {(1-confidence)*100:.2f}%</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
