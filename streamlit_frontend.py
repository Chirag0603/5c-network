import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

st.title('Brain MRI Metastasis Segmentation')

# File uploader for users to upload MRI images
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "png", "jpeg", "tif"])

# If a file is uploaded, send it to the backend FastAPI for prediction
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the uploaded file to bytes for sending it to the API
    bytes_data = uploaded_file.getvalue()

    # Call the FastAPI backend to get the prediction
    with st.spinner('Segmenting...'):
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": bytes_data})

    # Get the response content and display the segmented image
    if response.status_code == 200:
        st.success('Segmentation complete!')
        # Convert the response content
