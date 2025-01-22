# Plant-Disease-Prediction
A deep learning project to predict plant diseases using the PlantVillage dataset.
# Overview

This project is a **Plant Disease Prediction System** that uses a deep learning model to identify diseases in plants based on leaf images. The model is trained using the PlantVillage Dataset, which contains images of various crops and their associated diseases. The application provides actionable insights and treatment recommendations to mitigate risks, supporting farmers and agricultural professionals in managing crop health effectively.
## Model Interface
![Prediction Interface 2](Images/Prediction%20Interface%202.png)

The project includes a **Streamlit-based web interface** that allows users to upload an image of a plant leaf, classify the disease, and receive recommendations for treatment. The application supports multiple languages, including English and Hindi.

# Features

**Deep Learning Model:** Utilizes a pre-trained ResNet50 model for high accuracy.

**Data Augmentation:** Enhances model generalization by applying transformations like rotation, zoom, and flipping.

**Multi-language Support:** Provides disease names, affected crops, and treatment recommendations in English and Hindi.

**Streamlit Interface:** Easy-to-use web interface for image upload and disease classification.

**Actionable Insights:** Offers specific treatment recommendations and lists crops affected by the disease.

# Dataset

**Source**: PlantVillage Dataset

**Size:** Approximately 4GB

**Categories:** Contains images of healthy and diseased leaves for crops like apple, tomato, potato, and more.

**Note:** The dataset is not uploaded to GitHub due to its large size. Use the following API to download the dataset:

**import kagglehub
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")**

# Prerequisites

Ensure you have the following installed:

**Python 3.7+**

**TensorFlow**

**Streamlit**

**PIL (Pillow)**

**NumPy**

**Googletrans**

**Kaggle API**

# Model Architecture

## ResNet50

**Pre-trained on ImageNet:** The base model is frozen to retain learned features.

**Custom Layers:** Added a GlobalAveragePooling2D layer, Dense layers, and Dropout for classification.

## Training Details

**Image Size:** 224x224

**Batch Size:** 32

**Optimizer:** Adam with a learning rate of 0.0001

**Loss Function:** Categorical Crossentropy

**Early Stopping:** Monitors validation loss to prevent overfitting.

# Usage

Web Application
![Streamlit Interface](Images/Streamlit%20Interface.png)

Upload an image of a plant leaf.
![Image Input](Images/Image%20Input.png)

Select your preferred language (English or Hindi).

Click on the "Classify" button.

View the predicted disease, affected crops, and treatment recommendations.
![Prediction Interface](Images/Prediction%20Interface.png)


Example

Upload an image like PotatoHealthy1.JPG from the test set to see the prediction
![Prediction Interface 2](Images/Prediction%20Interface%202.png)

Billingual Supoort
![Prediction Interface Hindi](Images/Predicton%20Interface%20hindi.png)

# Files and Directories

**Dataset/:** Contains training and validation data (not included in this repository).

**model/:** Stores the trained model file (crop_disease_prediction_model.h5).

**app.py:** Streamlit application script.

**class_indices.json:** Maps class indices to disease names.

**requirements.txt:** Lists all dependencies.

