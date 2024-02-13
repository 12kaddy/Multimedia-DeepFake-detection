# app.py
from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from xgboost import XGBClassifier
import joblib

app = Flask(__name__)

# Load the XGBoost model
model_filename = 'FakeAudio.joblib'
loaded_model = joblib.load('FakeAudio.joblib')

# Define or import your feature extraction functions
# Placeholder: Define or import your feature extraction functions
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

def extract_features(file_path):
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

@app.route('/')
def index():
    return render_template('AudioIndex.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    audio_file = request.files['audioFile']

    # Save the uploaded file temporarily
    temp_path = 'temp_audio.wav'
    audio_file.save(temp_path)

    # Extract features
    features = extract_features(temp_path)

    # Make prediction
    prediction = loaded_model.predict(features)

    # Convert prediction to human-readable format
    result = "Fake" if prediction == 1 else "Real"

    # Remove the temporary file
    os.remove(temp_path)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
