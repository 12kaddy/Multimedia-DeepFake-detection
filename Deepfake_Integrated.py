from flask import Flask, render_template, request, jsonify, redirect, url_for
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

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Video Deepfake Detection
def extract_frames(video_path):
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

def make_predictions(frames):
    loaded_model = load_model('DF_model.h5')  # Replace 'your_model.h5' with your model file
    predictions = []
    for frame in frames:
        # Preprocess the frame according to your model's requirements
        processed_frame = frame  # Placeholder; add actual preprocessing steps
        # Make predictions using the loaded model
        prediction = loaded_model.predict(np.expand_dims(processed_frame, axis=0))
        predictions.append(prediction)
    return predictions

def is_deepfake(predictions, threshold=0.5):
    return any(pred[0][0] > threshold for pred in predictions)

# Audio Deepfake Detection
model_filename = 'FakeAudio.joblib'
loaded_model = joblib.load(model_filename)

def chroma_stft(data, sr):
    stft = np.abs(librosa.stft(data))
    chroma_stft_result = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.mean(np.squeeze(chroma_stft_result.T))

def rms(data):
    rms_result = librosa.feature.rms(y=data)
    return np.mean(np.squeeze(rms_result))

def spectral_centroid(data, sr):
    spectral_centroid_result = librosa.feature.spectral_centroid(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_centroid_result))

def spectral_bandwidth(data, sr):
    spectral_bandwidth_result = librosa.feature.spectral_bandwidth(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_bandwidth_result))

def spectral_rolloff(data, sr):
    spectral_rolloff_result = librosa.feature.spectral_rolloff(y=data, sr=sr)
    return np.mean(np.squeeze(spectral_rolloff_result))

def zero_crossing_rate(data):
    zero_crossing_rate_result = librosa.feature.zero_crossing_rate(y=data)
    return np.mean(np.squeeze(zero_crossing_rate_result))

def calculate_mfcc(data, sr, n_mfcc=20):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc_feature.T, axis=0)

def extract_features(file_path):
    data, sr = librosa.load(file_path, sr=None)
    chroma_stft_result = chroma_stft(data, sr)
    rms_result = rms(data)
    spectral_centroid_result = spectral_centroid(data, sr)
    spectral_bandwidth_result = spectral_bandwidth(data, sr)
    spectral_rolloff_result = spectral_rolloff(data, sr)
    zero_crossing_rate_result = zero_crossing_rate(data)
    mfcc_result = calculate_mfcc(data, sr)

    features = np.array([
        chroma_stft_result,
        rms_result,
        spectral_centroid_result,
        spectral_bandwidth_result,
        spectral_rolloff_result,
        zero_crossing_rate_result,
        *mfcc_result
    ])

    return features.reshape(1, -1)

# Image Deepfake Detection
model_image = load_model('Deepfake_image.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model_image.predict(img_array)
    heatmap_overlay = generate_heatmap_overlay(img_array, model_image, 'conv2d_5')
    heatmap_overlay_base64 = cv2_to_base64(heatmap_overlay)
    class_names = ['Real', 'Fake']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class, heatmap_overlay_base64

def generate_heatmap_overlay(img_array, model, last_conv_layer_name):
    # Implementation of heatmap generation
    # Return the heatmap overlay image
    return img_array  # Placeholder; add actual heatmap generation logic

def cv2_to_base64(img):
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def home():
    return render_template('deepfake2.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            return redirect(url_for('results', filename=file.filename))
    return render_template('sub.html')

@app.route('/results/<filename>')
def results(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if filename.endswith(('.mp3', '.wav')):
        result = detect_audio(file_path)
        return render_template('audioresult.html', result=result)
    elif filename.endswith(('.mp4', '.avi')):
        result = detect_video(file_path)
        return render_template('videoresult.html', result=result)
    elif filename.endswith(('.jpg', '.png')):
        result, heatmap_overlay_base64 = detect_image(file_path)
        return render_template('Imageresult.html', result=result, heatmap_overlay=heatmap_overlay_base64)
    else:
        return "Unsupported file format"

def detect_audio(file_path):
    data, sr = librosa.load(file_path, sr=None)
    features = extract_features(file_path)
    prediction = loaded_model.predict(features)
    result = "Fake" if prediction == 1 else "Real"
    return result

def detect_video(file_path):
    frames = extract_frames(file_path)
    predictions = make_predictions(frames)
    result = "This video contains deepfake content." if is_deepfake(predictions) else "This video is not identified as deepfake."
    return result

def detect_image(file_path):
    predicted_class, heatmap_overlay_base64 = predict_image(file_path)
    return predicted_class, heatmap_overlay_base64

if __name__ == '__main__':
    app.run(debug=True)
