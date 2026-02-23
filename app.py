import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp {
    background: radial-gradient(circle at top, #1a1a3d, #0a0a23 60%);
    color: white;
}

/* Title */
.main-title {
    text-align:center;
    font-size:60px;
    font-weight:800;
    background: linear-gradient(90deg,#00f5ff,#ff00c8);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.sub-text {
    text-align:center;
    font-size:22px;
    color:#cbd5e1;
    margin-bottom:40px;
}

/* Feature Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding:30px;
    border-radius:20px;
    backdrop-filter: blur(15px);
    text-align:center;
    transition:0.3s;
    border:1px solid rgba(255,255,255,0.1);
}
.card:hover {
    transform: translateY(-10px);
    box-shadow:0 0 25px #00f5ff;
}

/* Upload Box */
.upload-box {
    border:2px dashed #00f5ff;
    padding:40px;
    border-radius:20px;
    text-align:center;
    background: rgba(255,255,255,0.03);
}

/* Result Box */
.result-box {
    padding:25px;
    border-radius:15px;
    margin-top:20px;
    text-align:center;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="main-title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-Powered Pneumonia Detection from Chest X-Rays</div>', unsafe_allow_html=True)

# ---------- FEATURES ----------
st.markdown("### ✨ Key Features")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">⚡<h4>Fast Diagnosis</h4><p>Get results in seconds</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card">🎯<h4>Accurate Model</h4><p>Trained on thousands of images</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card">💻<h4>User Friendly</h4><p>Simple & elegant interface</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card">🧠<h4>AI Powered</h4><p>Advanced deep learning system</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- UPLOAD SECTION ----------
st.markdown("## 📤 Upload Your Chest X-Ray")

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    image_resized = image.resize((width, height))
    img_array = np.array(image_resized)

    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.uint8)

    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.image(image, width=300)

    st.markdown("## 🔍 Analysis Result")

    if confidence > 0.5:
        st.markdown(
            f'<div class="result-box" style="background:rgba(255,0,0,0.2); border:1px solid red;">'
            f'🔴 Pneumonia Detected<br>Confidence: {confidence*100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(int(confidence*100))
    else:
        st.markdown(
            f'<div class="result-box" style="background:rgba(0,255,100,0.2); border:1px solid #00ff88;">'
            f'🟢 No Pneumonia Detected<br>Confidence: {(1-confidence)*100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(int((1-confidence)*100))
