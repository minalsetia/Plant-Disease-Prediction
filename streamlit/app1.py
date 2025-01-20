import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Translate function
def translate_text(text, target_language):
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Specify the exact model path
model_path = r"D:/project 7th sem/minor project/model/crop_disease_prediction_model.h5"

# Load the pre-trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
else:
    print(f"Model not found at path: {model_path}")

class_indices_path = r"D:/project 7th sem/minor project/Dataset/class_indices.json"

# Load the class indices from the JSON file
if os.path.exists(class_indices_path):
    class_indices = json.load(open(class_indices_path))
    print("Class indices loaded successfully!")
else:
    print(f"Class indices file not found at path: {class_indices_path}")

# Disease to Crops Mapping (Diseases may affect multiple crops)
disease_mappings = {
    "Apple___Apple_scab": ["Apple"],
    "Apple___Black_rot": ["Apple", "Grape"],  # Black rot affects both Apple and Grape
    "Apple___Cedar_apple_rust": ["Apple"],
    "Blueberry___healthy": ["Blueberry"],
    "Cherry_(including_sour)___Powdery_mildew": ["Cherry"],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["Corn (maize)"],
    "Grape___Black_rot": ["Apple", "Grape"],  # Black rot affects both Apple and Grape
    "Orange___Haunglongbing_(Citrus_greening)": ["Orange"],
    "Peach___Bacterial_spot": ["Peach", "Tomato", "Pepper"],
    "Pepper,_bell___Bacterial_spot": ["Pepper", "Tomato", "Peach"],
    "Potato___Early_blight": ["Potato", "Tomato"],
    "Potato___Late_blight": ["Potato", "Tomato"],
    "Squash___Powdery_mildew": ["Squash", "Cherry"],
    "Tomato___Bacterial_spot": ["Tomato", "Pepper", "Peach"],
    "Tomato___Early_blight": ["Tomato", "Potato"],
    "Tomato___Late_blight": ["Tomato", "Potato"],
    "Tomato___healthy": ["Tomato"],
    "Tomato___Spider_mites Two-spotted_spider_mite": ["Tomato"]
}

# Recommendations dictionary
recommendations_dict = {
    "Apple___Apple_scab": "Apply fungicides containing mancozeb or myclobutanil.",
    "Apple___Black_rot": "Prune infected plant parts and apply copper-based fungicides.",
    "Apple___Cedar_apple_rust": "Apply fungicide containing myclobutanil.",
    "Blueberry___healthy": "Ensure proper pruning and irrigation for healthy growth.",
    "Cherry_(including_sour)___Powdery_mildew": "Use sulfur-based fungicides to treat powdery mildew.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides containing chlorothalonil or mancozeb.",
    "Grape___Black_rot": "Remove infected parts and use fungicides like captan or myclobutanil.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees and apply systemic insecticides.",
    "Peach___Bacterial_spot": "Use copper-based bactericides and remove infected leaves.",
    "Pepper,_bell___Bacterial_spot": "Use copper-based fungicides and remove infected leaves.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Use fungicides such as copper or metalaxyl, and avoid overwatering.",
    "Squash___Powdery_mildew": "Apply fungicides containing sulfur or potassium bicarbonate.",
    "Tomato___Bacterial_spot": "Apply copper-based fungicides and remove infected leaves.",
    "Tomato___Early_blight": "Use fungicides containing chlorothalonil or mancozeb.",
    "Tomato___Late_blight": "Apply fungicides like mancozeb and avoid overhead watering.",
    "Tomato___healthy": "Ensure proper spacing and avoid water stress.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Introduce predatory mites, use horticultural oils, and ensure proper irrigation to reduce mite infestations."
}


# Load and Preprocess Test Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Add language selection option
language = st.selectbox("Select Language", ["english", "hindi"])

# Define UI Texts
ui_texts = {
    "title": {
        "english": "Plant Disease Classifier",
        "hindi": "पौध रोग वर्गीकरण"
    },
    "upload_prompt": {
        "english": "Upload an image...",
        "hindi": "एक चित्र अपलोड करें..."
    },
    "classify_button": {
        "english": "Classify",
        "hindi": "वर्गीकृत करें"
    },
    "prediction_label": {
        "english": "Prediction: ",
        "hindi": "पूर्वानुमान: "
    },
    "recommendation_label": {
        "english": "Recommendation: ",
        "hindi": "सिफारिश: "
    },
    "affected_crops_label": {
        "english": "Affected Plants: ",
        "hindi": "प्रभावित पौधे : "
    }
}

# Translate all UI elements based on language selection
st.title(translate_text(ui_texts["title"][language], language))

uploaded_image = st.file_uploader(translate_text(ui_texts["upload_prompt"][language], language), type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button(translate_text(ui_texts["classify_button"][language], language)):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            
            # Get the crops associated with the predicted disease
            affected_crops = disease_mappings.get(prediction, ["Unknown Crop"])
            
            # Get the recommendation for the predicted disease
            recommendation = recommendations_dict.get(prediction, "No specific recommendation available.")
            
            # Translate the disease name, crops, and recommendation based on the selected language
            translated_disease = translate_text(prediction, 'hi' if language == "hindi" else 'en')
            translated_crops = ', '.join([translate_text(crop, 'hi' if language == "hindi" else 'en') for crop in affected_crops])
            translated_recommendation = translate_text(recommendation, 'hi' if language == "hindi" else 'en')
            
            # Show translated results
            st.success(f'{translate_text(ui_texts["prediction_label"][language], language)} {translated_disease}')
            st.info(f'{translate_text(ui_texts["recommendation_label"][language], language)} {translated_recommendation}')
            st.write(f'{translate_text(ui_texts["affected_crops_label"][language], language)} {translated_crops}')
