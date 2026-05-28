import streamlit as st
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model

# Load the saved best model
# Ensure the model file 'best_cnn_model_cifar10.keras' exists in your Colab environment
try:
    loaded_model = load_model('best_cnn_model_cifar10.keras')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best_cnn_model_cifar10.keras' is in the current directory.")
    st.stop() # Stop the app if model can't be loaded

# Define the class names (as used during training)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file as bytes
    image_bytes = uploaded_file.read()

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Preprocess the image:
        # 1. Resize to (32, 32) as expected by the model
        # 2. Convert to numpy array
        image_resized = image.resize((32, 32))
        image_array = np.array(image_resized)

        # Ensure the image has 3 channels if it's grayscale or has an alpha channel
        if image_array.ndim == 2: # Grayscale
            image_array = np.stack((image_array,)*3, axis=-1)
        elif image_array.shape[-1] == 4: # RGBA
            image_array = image_array[..., :3] # Take only RGB channels

        # Expand dimensions to match model's input shape (batch_size, 32, 32, 3)
        image_for_prediction = np.expand_dims(image_array, axis=0)

        # Make prediction
        prediction = loaded_model.predict(image_for_prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100

        st.success(f"Prediction: **{predicted_class_name}** with **{confidence:.2f}%** confidence")

    except Exception as e:
        st.error(f"Error processing image for prediction: {e}")


