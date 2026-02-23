import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- NAVIGATION ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_upload():
    st.session_state.page = "upload"

# ---------- CSS ----------
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg,#dbe6f6,#fef6fb); text-align:center; }
button { font-size:18px; padding:10px 30px; margin:10px; border-radius:12px; cursor:pointer; border:none; background:#4f9aff; color:white; transition:0.3s; }
button:hover { background:#3777e6; }
.upload-card { background: rgba(255,255,255,0.25); backdrop-filter: blur(12px); border-radius:30px; padding:50px; max-width:500px; margin:0 auto; box-shadow:0 8px 30px rgba(0,0,0,0.1);}
.result-text { font-size:28px; font-weight:bold; margin-top:20px; }
.result-normal { color:#22c55e; }
.result-pneumonia { color:#ef4444; }
.confidence-bar { height:20px; border-radius:10px; background:linear-gradient(to right,#22c55e,#ef4444); margin:15px auto;}
</style>
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

# ---------- HOME PAGE ----------
if st.session_state.page == "home":
    st.markdown('<h1 style="font-size:64px; font-weight:900;">🫁 PneumoAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:22px; color:#555;">AI-Powered Pneumonia Detection Instantly</p>', unsafe_allow_html=True)
    st.markdown('<button onclick="window.location.reload();">Start Diagnosis</button>', unsafe_allow_html=True)
    st.button("Go to Upload Page", on_click=go_upload)

# ---------- UPLOAD PAGE ----------
elif st.session_state.page == "upload":
    st.button("🏠 Back to Home", on_click=go_home)
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Your Chest X-Ray", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)

        img = image.resize((150,150))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        with st.spinner("Analyzing X-Ray..."):
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

        confidence = float(prediction[0][0])
        if confidence > 0.5:
            st.markdown(f"<div class='result-text result-pneumonia'>🔴 Pneumonia Detected</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-bar' style='width:{confidence*100}%;'></div>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence*100:.2f}%")
        else:
            st.markdown(f"<div class='result-text result-normal'>🟢 Normal Chest X-Ray</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-bar' style='width:{(1-confidence)*100}%;'></div>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {(1-confidence)*100:.2f}%")
    else:
        st.info("Upload an X-ray image to get diagnosis")
    st.markdown('</div>', unsafe_allow_html=True)
