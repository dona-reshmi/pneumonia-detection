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
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4 {
    font-weight: 800;
}

.hero {
    text-align: center;
    padding: 80px 20px 50px 20px;
}

.hero-title {
    font-size: 70px;
    margin-bottom: 20px;
}

.hero-subtitle {
    font-size: 24px;
    color: #cbd5f5;
    margin-bottom: 40px;
}

.section {
    padding: 60px 40px;
    text-align: center;
}

.card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    margin: 20px auto;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
    max-width: 500px;
}

.upload-box {
    border: 2px dashed #93c5fd;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
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
    gap: 20px;
}

.feature-card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    width: 250px;
    box-shadow: 0 0 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Pneumonia Detection at Your Fingertips</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- ABOUT ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("## About PneumoAI")
st.markdown("PneumoAI is a deep learning-based AI system designed to **detect pneumonia from chest X-rays** accurately and quickly. Perfect for educational use and early detection simulations.")
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

    if confidence > 0.5:
        st.markdown(f"<div class='result-pneumonia'>🔴 Pneumonia Detected</div>", unsafe_allow_html=True)
        st.progress(int(confidence*100))
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.markdown(f"<div class='result-normal'>🟢 Normal Chest X-Ray</div>", unsafe_allow_html=True)
        st.progress(int((1-confidence)*100))
        st.write(f"**Confidence:** {(1-confidence)*100:.2f}%")
else:
    st.info("Upload an X-ray image to get diagnosis")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURES ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("## 🌟 Features")

st.markdown('<div class="features">', unsafe_allow_html=True)
feature_list = [
    ("⚡ Fast Diagnosis", "Get results within seconds using AI"),
    ("🎯 High Accuracy", "Trained on real chest X-ray data"),
    ("💻 User-Friendly", "Simple and elegant interface"),
    ("🧠 AI-Powered", "Deep learning-based predictions"),
]

for icon, desc in feature_list:
    st.markdown(f'<div class="feature-card"><h3>{icon}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("<hr>")
st.markdown("<center>🧠 Developed with Deep Learning | 🏥 For Educational Use Only</center>")
st.markdown('</div>', unsafe_allow_html=True)
