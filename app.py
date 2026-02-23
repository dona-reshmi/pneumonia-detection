import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", page_icon="🫁", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #dbe6f6, #fef6fb);
    margin: 0; padding: 0;
    color: #111;
}

/* HERO */
.hero {
    text-align: center;
    padding: 120px 20px 80px 20px;
    animation: float 6s ease-in-out infinite;
}
.hero-title {
    font-size: 70px;
    font-weight: 900;
    text-shadow: 2px 2px 12px rgba(0,0,0,0.15);
}
.hero-subtitle {
    font-size: 24px;
    margin-bottom: 50px;
    color: #555;
}

/* FLOAT ANIMATION */
@keyframes float {
    0% { transform: translateY(0px);}
    50% { transform: translateY(-10px);}
    100% { transform: translateY(0px);}
}

/* SECTIONS */
.section { padding: 60px 20px; text-align: center; }
.section-title { font-size: 36px; font-weight: 700; margin-bottom: 40px; }

/* NEUMORPH CARDS */
.card {
    background: #e0e5ec;
    border-radius: 25px;
    padding: 30px;
    margin: 10px;
    width: 220px;
    display: inline-block;
    cursor: pointer;
    transition: transform 0.4s, box-shadow 0.4s;
    box-shadow: 8px 8px 20px #babecc, -8px -8px 20px #ffffff;
}
.card:hover {
    transform: translateY(-10px);
    box-shadow: 4px 4px 15px #babecc, -4px -4px 15px #ffffff;
}

/* GLASS UPLOAD CARD */
.upload-card {
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(12px);
    border-radius: 30px;
    padding: 50px;
    max-width: 500px;
    margin: 0 auto;
    box-shadow: 0 8px 30px rgba(0,0,0,0.1);
}

/* RESULT TEXT */
.result-text { font-size: 28px; font-weight: bold; margin-top: 15px; }
.result-normal { color: #22c55e; }
.result-pneumonia { color: #ef4444; }

/* CONFIDENCE BAR */
.confidence-bar {
    height: 25px; 
    border-radius: 12px; 
    background: linear-gradient(to right, #22c55e, #ef4444);
    margin: 15px auto;
    width: 50%;
}

/* FOOTER */
.footer { text-align: center; padding: 20px 20px; font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🫁 PneumoAI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Pneumonia Detection Instantly</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURES ----------
st.markdown('<div class="section" id="features">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌟 Features</div>', unsafe_allow_html=True)

features = [
    ("⚡ Fast Diagnosis", "Get results in seconds", "#upload"),
    ("🎯 High Accuracy", "Trained on real chest X-ray data", "#upload"),
    ("💻 Sleek UI", "Smooth, modern interface", "#upload"),
    ("🧠 AI-Powered", "Deep learning-based predictions", "#upload")
]

for title, desc, link in features:
    st.markdown(f'<a href="{link}" style="text-decoration:none;"><div class="card"><h3>{title}</h3><p>{desc}</p></div></a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
st.markdown('<div class="section" id="upload">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📤 Upload Your Chest X-Ray</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"])

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("🧠 Developed with Deep Learning | 🏥 For Educational Use Only")
st.markdown('</div>', unsafe_allow_html=True)
