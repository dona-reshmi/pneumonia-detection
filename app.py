import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PneumoAI", layout="centered")

st.title("🫁 Pneumonia Detection App")

# Load TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="pneumonia_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Get expected input shape from model
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    image = image.resize((width, height))
    img_array = np.array(image)

    # Adjust dtype automatically
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.uint8)

    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        st.error(f"🔴 Pneumonia Detected (Confidence: {confidence*100:.2f}%)")
    else:
        st.success(f"🟢 Healthy (Confidence: {(1-confidence)*100:.2f}%)")
