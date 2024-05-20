import streamlit as st
from PIL import Image
import numpy as np
import time
import cv2
from skimage.transform import resize
from keras.models import load_model
import threading

# Define a function to load the model and use session state
@st.cache_resource
def load_model_once():
    model_path = r'models/DenseNet_model.h5'
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {e}")
        model = None
    return model

# Histogram equalization
def histogram_equalization(image):
    modified_img = image.copy()
    modified_img = cv2.equalizeHist(modified_img)
    return modified_img

def minimize_gray_noise(image):
    gray_threshold = 125
    modified_img = image.copy()
    for i in range(modified_img.shape[0]):
        for j in range(modified_img.shape[1]):
            pixel = modified_img[i, j]
            if pixel < gray_threshold:
                modified_img[i, j] = 0
    return modified_img

def inpaint_image(image):
    modified_img = image.copy()
    _, mask = cv2.threshold(modified_img, 230, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(modified_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Function to process and predict the image
def process_and_predict(image_array, model):
    try:
        image_equalize = histogram_equalization(image_array)
        image_equalize = inpaint_image(image_equalize)
        col2.image(image_equalize, width=300, use_column_width=True, clamp=True, caption='HISTOGRAM EQUALIZATION & INPAINT')

        image_equalize = minimize_gray_noise(image_equalize)
        col3.image(image_equalize, width=300, use_column_width=True, clamp=True, caption='NOISE REMOVAL')
        image_equalize = resize(image_equalize, (224, 224))

        if len(image_equalize.shape) == 2:
            image_equalize = np.stack((image_equalize,) * 3, axis=-1)

        st.write(f"Input image shape before adding batch dimension: {image_equalize.shape}")
        image_data_with_batch = np.expand_dims(image_equalize, axis=0)
        st.write(f"Input image shape after adding batch dimension: {image_data_with_batch.shape}")

        image_data_with_batch = image_data_with_batch.astype(np.float32)

        try:
            predictions = model.predict(image_data_with_batch)
            st.write(f"Predictions: {predictions}")
            if predictions < 0.5:
                st.markdown("<span style='color:red; font-size:30px; font-weight: bold; background-color: white'>FRACTURED</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green; font-size:30px; font-weight: bold; background-color: white'>NORMAL</span>", unsafe_allow_html=True)
        except Exception as e:
            st.write(f"Error during prediction: {e}")

    except Exception as e:
        st.write(f"Error during preprocessing: {e}")

# Project Title
st.title("Deep-Learning Based Binary Classification of Wrist Fracture")

# Allow the user to upload an image
user_image = st.file_uploader("Upload an image", ['png', 'jpg', 'jpeg', 'tif'])

# Define columns for layout
col1, col2, col3 = st.columns(3)

if user_image is not None:
    image_pil = Image.open(user_image)
    image_array = np.array(image_pil)
    if len(image_array.shape) != 2:
        image_array = np.mean(image_array, axis=2)

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    if min_val < 0 or max_val > 255:
        image_array = ((image_array - min_val) / (max_val - min_val)) * 255
        image_array = image_array.astype(np.uint8)

    if image_array.dtype != "uint8":
        image_array = image_array.astype(np.uint8)

    col1.image(image_array, width=300, use_column_width=True, clamp=True, caption='ORIGINAL IMAGE')

progress_text = "PREDICTING PLEASE WAIT..."

if st.button("PREDICT"):
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)

    model = load_model_once()

    if model is not None:
        prediction_thread = threading.Thread(target=process_and_predict, args=(image_array, model))
        prediction_thread.start()
        prediction_thread.join()

    my_bar.empty()
