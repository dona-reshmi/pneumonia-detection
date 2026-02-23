import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="centered")

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.markdown("Upload a chest X-ray image to predict whether the patient has Pneumonia or Normal.")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150,150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    with st.spinner("🔍 Analyzing Image... Please wait"):
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    st.subheader("📊 Prediction Result")

    if confidence > 0.5:
        st.error(f"🛑 Pneumonia Detected")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.success("✅ Normal (No Pneumonia)")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")

    st.progress(float(confidence))


# Sidebar
st.sidebar.header("ℹ️ About This Project")
st.sidebar.write("""
This is a Deep Learning based Pneumonia Detection system 
built using a Convolutional Neural Network (CNN) model 
and deployed using Streamlit Cloud.
""")

st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 Developed by Dona Reshmi")
st.sidebar.write("🚀 Deployed using Streamlit")
