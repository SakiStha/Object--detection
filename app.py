import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image  # Use Pillow instead of OpenCV for image loading

st.title("YOLOv8 Object Detection")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Image uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Read image with Pillow (no OpenCV)
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert to numpy array

    # Run YOLOv8 inference
    results = model(image_np)

    # Display results
    st.image(
        results[0].plot(),  # Still uses OpenCV internally, but HEADLESS works
        caption="Detected Objects",
        use_column_width=True
    )
