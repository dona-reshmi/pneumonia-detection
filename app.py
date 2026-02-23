import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("pneumonia_model.h5")

st.title("Pneumonia Detection App")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        st.write("Prediction: Pneumonia")
    else:
        st.write("Prediction: Healthy")
