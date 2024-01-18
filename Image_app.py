from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import base64

app = Flask(__name__)

# Load your trained model
model = load_model('Deepfake_image.h5')

# Function to preprocess and predict an image
def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Display heatmap overlay
    heatmap_overlay = generate_heatmap_overlay(img_array, model, 'conv2d_5')

    # Convert heatmap overlay to base64 for embedding in HTML
    heatmap_overlay_base64 = cv2_to_base64(heatmap_overlay)

    # Interpret the prediction (modify this based on your class labels)
    class_names = ['Real', 'Fake']
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class, heatmap_overlay_base64

# Function to generate heatmap overlay
def generate_heatmap_overlay(img_array, model, last_conv_layer_name):
    # ... (your heatmap generation code)

    # Return the heatmap overlay image
    return img_heatmap_overlay

# Function to convert OpenCV image to base64
def cv2_to_base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# Flask route for the main page
@app.route('/')
def index():
    return render_template('index2.html')

# Flask route for handling file upload and displaying results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index2.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index2.html', error='No selected file')

    # Save the uploaded image to a temporary file
    temp_image_path = 'input/Temp/D_image.jpg'
    file.save(temp_image_path)

    # Get predictions and heatmap overlay
    predicted_class, heatmap_overlay_base64 = predict_image(temp_image_path)

    # Render the result page with predictions and heatmap overlay
    return render_template('result2.html', predicted_class=predicted_class, heatmap_overlay=heatmap_overlay_base64)

if __name__ == '__main__':
    app.run(debug=True)
