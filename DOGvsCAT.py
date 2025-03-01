import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("cat_dog_classifier.h5")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # 70% for strong classification

# Streamlit UI
st.title("Cat vs. Dog Classifier")
st.write("Upload an image or use your webcam to classify whether it's a **Cat** or a **Dog**!")

# Select input method
option = st.radio("Choose Input Method:", ("Upload Image", "Use Webcam"))

def preprocess_image(image):
    """Preprocess the image to match model input shape."""
    img_array = np.array(image)
    
    # Ensure 3 channels
    if img_array.shape[-1] == 4:  # If PNG has transparency (RGBA)
        img_array = img_array[:, :, :3]

    img_resized = cv2.resize(img_array, (128, 128))  # Resize to 128x128
    img_resized = img_resized / 255.0  # Normalize (0-1)
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_resized

def classify_image(image):
    """Predict whether the image is Dog, Cat, or Neither."""
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    probability = prediction[0][0]

    # Classification logic
    if probability > CONFIDENCE_THRESHOLD:  
        return "🐶 Dog", probability
    elif probability < (1 - CONFIDENCE_THRESHOLD):  
        return "🐱 Cat", probability
    else:  
        return "❌ Neither Dog Nor Cat", probability

# 📷 Option 1: Upload Image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        label, prob = classify_image(image)

        # Show result
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write(f"### Prediction: {label} (Confidence: {prob:.2f})")

# 📸 Option 2: Use Webcam
elif option == "Use Webcam":
    cam_img = st.camera_input("Take a picture")

    if cam_img is not None:
        image = Image.open(cam_img)
        label, prob = classify_image(image)

        # Show result
        st.image(image, caption="Captured Image", use_container_width=True)
        st.write(f"### Prediction: {label} (Confidence: {prob:.2f})")
