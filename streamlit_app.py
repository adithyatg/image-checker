import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Page title
st.set_page_config(page_title='Image Classification App', page_icon='🖼️')
st.title('🖼️ Image Classification App')

# Sidebar for accepting input image
with st.sidebar:
    st.header('Upload Image')
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('https://github.com/username/repository/raw/main/your_model.h5')
    return model

model = load_model()

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

# Display the uploaded image and prediction result
if uploaded_image is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Predict the image class
    prediction = predict(img)
    
    # Display the prediction result
    st.subheader('Prediction Result:')
    st.write(prediction)
