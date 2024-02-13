from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    loaded_model = load_model('your_model.keras')  # Replace 'your_model.h5' with your model file
    predictions = []
    for frame in frames:
        # Preprocess the frame according to your model's requirements
        processed_frame = frame  # Placeholder; add actual preprocessing steps
        # Make predictions using the loaded model
        prediction = loaded_model.predict(np.expand_dims(processed_frame, axis=0))
        predictions.append(prediction)
    return predictions

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/extract_frames', methods=['POST'])
def extract_and_predict():
    if 'video' not in request.files:
        return 'No video file found'

    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    frames = extract_frames(video_path)
    predictions = make_predictions(frames)
    # Process predictions as needed
    # Example: convert predictions to a user-readable format
    formatted_predictions = [f"Frame {i}: {pred}" for i, pred in enumerate(predictions)]
    return render_template('predictions.html', predictions=formatted_predictions)

if __name__ == '__main__':
    app.run(debug=True)
