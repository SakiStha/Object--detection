import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLOv5 model from Ultralytics (pretrained on COCO)
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect(image, model):
    # Convert PIL image to numpy array
    img = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
   
    # Run detection
    results = model(img)
   
    # Render results on the image
    results.render()  # updates results.imgs with boxes and labels
   
    # Convert back to RGB for display in Streamlit
    detected_img = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)
    return detected_img, results

def main():
    st.title("YOLOv5 Object Detection with Streamlit")
    st.write("Upload an image and get object detection results using YOLOv5")
   
    model = load_model()
   
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
       
        if st.button('Detect Objects'):
            with st.spinner('Running YOLOv5...'):
                detected_img, results = detect(image, model)
           
            st.image(detected_img, caption='Detected Objects', use_column_width=True)
            st.write("Detections:")
            st.write(results.pandas().xyxy[0])  # pandas dataframe of detections

if __name__ == "__main__":
    main()
