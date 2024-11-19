import sys
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Add the project path
sys.path.append(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection')
from braintumordetection.modeling.model import testdf

def preprocess_image(image_path, target_size=(240, 240)):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Convert the image to an array
    image = img_to_array(image)
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=0)
    # Normalize the image
    image = image / 255.0
    return image

def modelpred(model_path, modelname, image_path):
    model_path = os.path.join(model_path, modelname)
    model = load_model(model_path)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(image)
    pred_ = np.argmax(prediction, axis=1)[0]

    labels = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'normal', 3: 'pituitary_tumor'}

    if pred_ in labels.keys():
        return labels[pred_]
