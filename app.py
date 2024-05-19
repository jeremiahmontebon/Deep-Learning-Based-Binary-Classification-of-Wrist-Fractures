import streamlit as st
from PIL import Image
import numpy as np
import time
import pandas as pd
import cv2
from skimage.transform import resize
from keras.models import load_model

# Histogram equalization
def histogram_equalization(image):
    # Create a copy of the image to avoid modifying the original image
    modified_img = image.copy()
    # Convert the image to grayscale if it's not already in grayscale

    modified_img = cv2.equalizeHist(modified_img)

    return modified_img

def minimize_gray_noise(image):
    gray_threshold = 125
    # Create a copy of the image to avoid modifying the original image
    modified_img = image.copy()
    # Iterate through each pixel and change gray colors to black
    for i in range(modified_img.shape[0]):
        for j in range(modified_img.shape[1]):
            pixel = modified_img[i, j]
            if (pixel < 125):
                modified_img[i, j] = 0
    return modified_img

# Inpainting
def inpaint_image(image):
    # Create a copy of the image
    modified_img = image.copy()
    #
    _, mask = cv2.threshold(modified_img, 230, 255, cv2.THRESH_BINARY)
    # Inpaint the letters using the mask
    inpainted_image = cv2.inpaint(modified_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Project Title
st.title("Deep-Learning Based Binary Classification of Wrist Fracture")

option = st.selectbox(
    "CHOOSE A MODEL",
    ("InceptionV3", "Resnet-50", "Densenet-201"))

# Allow the user to upload an image
user_image = st.file_uploader("Upload an image", ['png', 'jpg', 'jpeg', 'tif'])

# Define columns for layout
col1, col2, col3 = st.columns(3)


if user_image is not None:
    # Read the image using PIL
    image_pil = Image.open(user_image)
    image_array = np.array(image_pil)
    # Change shape to none/1
    if(len(image_array.shape) != 2):
        image_array = np.mean(image_array, axis=2)

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    if (min_val < 0 or max_val > 255):
        image_array = ((image_array - min_val) / (max_val - min_val)) * 255
        # Convert pixel values to uint8 data type
        image_array = image_array.astype(np.uint8)
    
    # Re-check those who abide with 0-255 and not uint8
    if(image_array.dtype != "uint8"):
        image_array = image_array.astype(np.uint8)
        
    # Display the uploaded image
    col1.image(image_array, width=300, use_column_width=True, clamp=True, caption='ORIGINAL IMAGE')

# Add button for prediction
progress_text = "PREDICTING PLEASE WAIT..."

# Create a button to trigger the prediction
if st.button("PREDICT"):
    # Create a progress bar
    my_bar = st.progress(0, text = progress_text)

    # Update the progress bar
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, progress_text)
        
    
    # Load the model from the HDF5 file
    if option == "InceptionV3":
        model = load_model('models\DenseNet_model.h5')
    elif option == "Resnet-50":
        model = load_model('models\DenseNet_model.h5')
    elif option == "Densenet-201":
        model = load_model('models\DenseNet_model.h5')
    else:
        # No model selected
        st.warning("Please select a model.")
    
    # Call pre-processing functions
    time.sleep(3)
    image_equalize = histogram_equalization(image_array)
    image_equalize = inpaint_image(image_equalize)
    col2.image(image_equalize, width=300, use_column_width=True, clamp=True, caption='HISTOGRAM EQUALIZATION & INPLACE')
    time.sleep(3)
    image_equalize = minimize_gray_noise(image_equalize)
    # Display the uploaded image
    col3.image(image_equalize, width=300, use_column_width=True, clamp=True, caption='NOISE REMOVAL')
    image_equalize = resize(image_equalize, (224, 224))
    
    # Reshape if channel is not 3/RGB 
    if(len(image_equalize.shape) == 2):
        image_equalize = np.stack((image_equalize,) * 3, axis=-1)
    
    # Add batch dimension
    image_data_with_batch = np.expand_dims(image_equalize, axis=0)
    predictions = model.predict(image_data_with_batch)
    if(predictions < 0.5):
        st.markdown("<span style='color:red; font-size:30px; font-weight: bold; background-color: white'>FRACTURED</span>",unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green; font-size:30px; font-weight: bold; background-color: white'>NORMAL</span>",unsafe_allow_html=True)

    # Add a delay before clearing the progress bar
    time.sleep(1)
    
    # Clear the progress bar
    my_bar.empty()