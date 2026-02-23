import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pneumonia AI", page_icon="🩺", layout="centered")

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #1f2937;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Cover Section ----------
st.markdown('<div class="title">🩺 AI Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a Chest X-ray image and let AI analyze it instantly.</div>', unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Upload Section ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("Analyzing with AI..."):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.subheader("Prediction Result")

    if confidence > 0.5:
        st.error("Pneumonia Detected")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.success("Normal")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)

