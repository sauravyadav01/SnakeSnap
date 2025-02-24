import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# -------------------------
# Helper Functions
# -------------------------
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the uploaded image for model prediction."""
    image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)
    return image

def predict_species(interpreter, class_names, image):
    """Return the predicted snake species and prediction confidence using TFLite."""
    processed_image = preprocess_image(image)
    
    # Get input and output details from the interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get the predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_species = class_names[predicted_idx]
    return predicted_species, confidence

# -------------------------
# Model Loader (cached using st.cache_resource)
# -------------------------
@st.cache_resource
def load_model():
    """
    Load the trained TensorFlow Lite model and return the interpreter along with its class names.
    Ensure 'snake_detector.tflite' is in the same directory.
    """
    interpreter = tf.lite.Interpreter(model_path="snake_detector.tflite")
    interpreter.allocate_tensors()
    # List of snake species in the same order as your model's output.
    class_names = [
        "Black Headed Royal Snake",
        "Cobra",
        "Common Krait",
        "Common Trinket",
        "Indian Boa",
        "Indian Cat",
        "Keelback",
        "Kukri",
        "Pit Vipper",
        "Python",
        "Racer Snake",
        "Rat Snake",
        "Russell Vipper",
        "Saw Scaled Vipper",
        "Wolf Snake"
    ]
    return interpreter, class_names

# Load the TFLite interpreter and class names (cached for performance)
interpreter, class_names = load_model()

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Predict"])

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("🐍 Welcome to the Snake Species Detector")
    st.markdown(
        """
        ### About the App
        This application uses a deep learning model to identify the species of snakes from images.
        Use the sidebar to navigate to the **Upload & Predict** page to test the model with your own snake images.
        """
    )
    
    # Load and display a Lottie animation for a dynamic landing page.
    lottie_snake = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_tfb3estd.json")
    if lottie_snake:
        st_lottie(lottie_snake, height=300)
    else:
        st.write("Animation failed to load. Please check the URL or your internet connection.")

    st.markdown("**Tip:** Use the sidebar to navigate to the prediction page.")

# -------------------------
# Upload & Predict Page
# -------------------------
elif page == "Upload & Predict":
    st.title("🖼️ Upload a Snake Image")
    st.markdown("Upload an image of a snake below to get a prediction of its species.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        st.write("Classifying the image...")
        with st.spinner("Predicting..."):
            species, confidence = predict_species(interpreter, class_names, image)
            st.success(f"Predicted Species: **{species}**")
            st.info(f"Confidence: **{confidence:.2f}**")
        
        # Optionally, display an animation after prediction.
        result_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_jbrw3hcz.json")
        if result_animation:
            st_lottie(result_animation, height=200)
        else:
            st.write("Result animation could not be loaded.")
    else:
        st.warning("Please upload an image to get started.")
