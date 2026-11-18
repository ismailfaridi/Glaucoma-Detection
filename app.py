import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Forces Keras to use the legacy loader for your .h5 model

st.set_page_config(page_title="Glaucoma Detector", page_icon="👁️", layout="centered")

# Load the model directly from the folder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/glaucoma_detector_EfficientNetB3_model.keras', compile=False)
    return model

model = load_model()

# Define CLAHE preprocessing (OpenCV)
def apply_clahe(img_array):
    img_uint8 = np.uint8(img_array)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img

# Streamlit UI
st.title("👁️ Glaucoma Detection System")
st.write("Upload a retinal fundus image to check for signs of Glaucoma using an EfficientNetB3 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner('Analyzing image...'):
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        
        # Apply CLAHE
        enhanced_array = apply_clahe(img_array)
        
        # Prepare tensor
        input_tensor = np.expand_dims(enhanced_array, axis=0)
        
        # Predict
        prob = model.predict(input_tensor, verbose=0)[0][0]
        
        # Display Results
        st.subheader("Diagnosis Result:")
        if prob >= 0.5:
            st.error(f"🛑 **Glaucoma Positive** (Confidence: {prob * 100:.2f}%)")
        else:
            st.success(f"✅ **Glaucoma Negative** (Confidence: {(1 - prob) * 100:.2f}%)")
        
        # Show the enhanced image for transparency
        # st.write("---")
        # st.write("**Preprocessed (CLAHE Enhanced) Image:**")
        # st.image(enhanced_array, caption='Enhanced for Model', use_container_width=True)