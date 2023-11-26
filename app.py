import streamlit as st
import pickle
import numpy as np 
import cv2
from sklearn.preprocessing import StandardScaler

# Load models
with open('face_detection.pkl', 'rb') as file:
    model = pickle.load(file)
with open('pca.pkl', 'rb') as file:  
    pca = pickle.load(file)

# Load and preprocess image
img_file = st.file_uploader("Upload an image")
if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE) 
    scaler = StandardScaler()
    img_std = scaler.fit_transform(img)

    # PCA and prediction
    img_pca = pca.transform(img_std.reshape(1,-1))
    pred = model.predict(img_pca)[0]
    
    # Display input image and predicted image side by side
    col1, col2 = st.columns(2)
    col1.image(img, use_column_width=True)
    predicted_image = cv2.imread(f"processed_faces/{pred}/{img_file.name}")
    #check if predicted image exists
    if predicted_image is None:
        predicted_image = cv2.imread(f"processed_faces/{pred}/{pred}_0001.jpg")
        col2.image(predicted_image, use_column_width=True)
    else:
        col2.image(predicted_image, use_column_width=True)
    
    # Display prediction
    st.write(f"Predicted name: {pred}")