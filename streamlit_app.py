import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

# Page title
st.set_page_config(page_title='Image Classification App', page_icon='🖼️')
st.title('🖼️ Image Classification App')

# Sidebar for accepting input image
with st.sidebar:
    st.header('Upload Image')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# URL of the model file
model_url = 'https://github.com/adithyatg/image-checker/raw/master/Jelly_Msand.h5'  # Use the raw URL for direct download

# Function to download the model file
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner('Downloading model...'):
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            st.success('Model downloaded successfully!')

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model(url):
    model_path = 'Jelly_Msand.h5'
    download_model(url, model_path)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model(model_url)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match model input size
    img = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(img)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make predictions
    prediction = model.predict(processed_image)
    return prediction

# Map predictions to class names
def map_prediction_to_class(prediction):
    class_names = ['Jelly', 'Msand']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, prediction[0][np.argmax(prediction)]

# Display the uploaded image and prediction result
if uploaded_image is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Predict the image class
    prediction = predict(img)
    predicted_class, probability = map_prediction_to_class(prediction)
    
    # Display the prediction result
    st.subheader('Prediction Result:')
    st.write(f'Class: {predicted_class}')
    st.write(f'Probability: {probability:.4f}')
