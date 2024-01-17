from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES

# from werkzeug.utils import secure_filename
# from werkzeug.utils import url_quote
import os
import cv2
import numpy as np
from keras.models import load_model
import librosa
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import base64

app = Flask(__name__)

# Configure file uploads
app.config['UPLOADED_FILES_DEST'] = 'uploads'
all_files = UploadSet('allfiles', IMAGES)
configure_uploads(app, allfiles)


# Load the XGBoost audio model
audio_model_filename = 'FakeAudio.joblib'
loaded_audio_model = joblib.load(audio_model_filename)

# Load the Keras video model (Replace with your actual video model file)
video_model_filename = 'your_model.keras'
loaded_video_model = load_model(video_model_filename)

# Load the Keras image model
image_model_filename = '/content/drive/My Drive/Deepfake/Deepfake_image.h5'
loaded_image_model = load_model(image_model_filename)

# Placeholder: Define or import your feature extraction functions for video
def extract_frames(video_path):
    # Implementation: Extract frames from video
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        resized_frame = cv2.resize(image, (128, 128))  # Resize frame to match model input size
        frames.append(resized_frame)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

def make_predictions_video(frames):
    # Implementati
    # on: Make predictions for video frames
    predictions = []
    for frame in frames:
        # Preprocess the frame according to your model's requirements
        processed_frame = frame  # Placeholder; add actual preprocessing steps
        # Make predictions using the loaded model
        prediction = loaded_video_model.predict(np.expand_dims(processed_frame, axis=0))
        predictions.append(prediction)
    return predictions

# Placeholder: Define or import other feature extraction functions for video

# Placeholder: Define or import functions to get labels or results based on predictions
def get_label_video(predictions):
    # Implementation: Get label for video predictions
    # Assuming the first element of the prediction is for class 0 (REAL)
    # You might need to adjust this based on your model's output
    real_confidence = predictions[0][0]

    # Set a threshold for confidence to determine if it's REAL or FAKE
    threshold = 0.5

    if real_confidence >= threshold:
        return "REAL"
    else:
        return "FAKE"

# Placeholder: Define or import your feature extraction functions for audio
def chroma_stft(data, sr):
    # Placeholder implementation: Calculate the mean of chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft_result = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.mean(np.squeeze(chroma_stft_result.T))

def rms(data):
    # Placeholder implementation: Calculate the mean of RMS
    rms_result = librosa.feature.rms(y=data)
    return np.mean(np.squeeze(rms_result))

def spectral_centroid(data, sr):
    # Placeholder implementation: Calculate the mean of spectral centroid
    spectral_centroid_result = librosa.feature.spectral_centroid(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_centroid_result))

def spectral_bandwidth(data, sr):
    # Placeholder implementation: Calculate the mean of spectral bandwidth
    spectral_bandwidth_result = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_bandwidth_result))

def spectral_rolloff(data, sr):
    # Placeholder implementation: Calculate the mean of spectral rolloff
    spectral_rolloff_result = librosa.feature.spectral_rolloff(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_rolloff_result))

def zero_crossing_rate(data):
    # Placeholder implementation: Calculate the mean of zero-crossing rate
    zero_crossing_rate_result = librosa.feature.zero_crossing_rate(y=data)
    return np.mean(np.squeeze(zero_crossing_rate_result))

def calculate_mfcc(data, sr, n_mfcc=20):
    # Placeholder implementation: Calculate the mean of MFCC coefficients
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc_feature.T, axis=0)

# ... (define or import other feature extraction functions)

def extract_features_audio(file_path):
    # Load the audio file
    data, sr = librosa.load(file_path, sr=None)

    # Extract features using your existing feature extraction functions
    chroma_stft_result = chroma_stft(data, sr)
    rms_result = rms(data)
    spectral_centroid_result = spectral_centroid(data, sr)
    spectral_bandwidth_result = spectral_bandwidth(data, sr)
    spectral_rolloff_result = spectral_rolloff(data, sr)
    zero_crossing_rate_result = zero_crossing_rate(data)
    mfcc_result = calculate_mfcc(data, sr)

    # Combine all features into a numpy array
    features = np.array([
        chroma_stft_result,
        rms_result,
        spectral_centroid_result,
        spectral_bandwidth_result,
        spectral_rolloff_result,
        zero_crossing_rate_result,
        *mfcc_result
    ])

    return features.reshape(1, -1)  # Reshape to (1, n_features) for model prediction

# ... (Define or import other feature extraction functions for audio)

# Placeholder: Define or import other functions needed for getting labels or results
def get_result_audio(prediction):
    # Implementation: Get result for audio prediction
    return "Fake" if prediction == 1 else "Real"


def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = loaded_image_model.predict(img_array)

    # Display heatmap overlay
    heatmap_overlay = generate_heatmap_overlay(img_array, loaded_image_model, 'conv2d_5')

    # Convert heatmap overlay to base64 for embedding in HTML
    heatmap_overlay_base64 = cv2_to_base64(heatmap_overlay)

    # Interpret the prediction (modify this based on your class labels)
    class_names = ['Real', 'Fake']
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class, heatmap_overlay_base64

def generate_heatmap_overlay(img_array, model, last_conv_layer_name):
    # ... (your heatmap generation code)

    # Return the heatmap overlay image
    return img_heatmap_overlay

def cv2_to_base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_files():
    if 'file' not in request.files:
        return 'No file found'

    uploaded_file = request.files['file']
    file_path = os.path.join(app.config['UPLOADED_FILES_DEST'], uploaded_file.filename)
    uploaded_file.save(file_path)

    if file_path.endswith('.wav'):
        # Process audio file
        audio_features = extract_features_audio(file_path)
        audio_prediction = loaded_audio_model.predict(audio_features)
        audio_result = get_result_audio(audio_prediction)
        return jsonify({'result': audio_result})

    elif file_path.endswith(('.mp4', '.avi', '.mkv')):
        # Process video file
        video_frames = extract_frames(file_path)
        video_predictions = make_predictions_video(video_frames)
        video_label = get_label_video(video_predictions)
        return jsonify({'label': video_label})

    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        # Process image file
        image_prediction = process_image(file_path)
        image_result = get_result_image(image_prediction)
        return jsonify({'result': image_result})

    else:
        return 'Unsupported file format'

if __name__ == '__main__':
    app.run(debug=True)
