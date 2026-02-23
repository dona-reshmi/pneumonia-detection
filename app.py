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

.stApp {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    text-align: center;
    font-size: 60px;
    font-weight: 800;
    margin-top: 40px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    margin-bottom: 40px;
    color: #cbd5f5;
}

.card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 0 25px rgba(0,0,0,0.3);
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
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown('<div class="title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Pneumonia Detection using Deep Learning</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- LAYOUT ----------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 AI Diagnosis Result")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=280)

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

# ---------- FOOTER ----------
st.markdown("""
<hr>
<center>
🧠 Developed with Deep Learning | 🏥 For Educational Use Only  
</center>
""", unsafe_allow_html=True)
