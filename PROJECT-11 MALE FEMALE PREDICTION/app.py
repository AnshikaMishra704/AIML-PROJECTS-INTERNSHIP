
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Male/Female Image Classifier", page_icon="🧑")

st.title("🧑 Male/Female Image Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("binary_image_classifier.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        label = "Male"
        confidence = prediction
    else:
        label = "Female"
        confidence = 1 - prediction

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
