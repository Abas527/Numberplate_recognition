#UI
import streamlit as st
import cv2
import numpy as np
from src.pipeline import detect_and_read

st.set_page_config(page_title="Nepali ANPR", layout="wide")

st.title(" Nepali Number Plate Recognition")
st.write("Upload an image to detect Nepali number plates.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width="stretch")

    with st.spinner("Detecting..."):
        result_img, texts = detect_and_read(img)

    st.image(result_img, caption="Detected Result", width="stretch")

    if texts:
        st.success("Detected Plate Numbers:")
        for t in texts:
            st.write(f"➡ {t}")
    else:
        st.warning("No plate detected.")