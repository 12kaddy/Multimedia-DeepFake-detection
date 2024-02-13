# RJPOLICE_HACK_569_Mobcipher_Problem-Statement-8

### Description:
This branch is dedicated to the development of advanced deep fake detection features within the overarching deep fake detection project. It encompasses the integration of multimodal analysis, real-time anomaly detection, and the incorporation of MesoNet for enhanced identification.

### Key Tasks

Integration of Video, Image, and Audio Analysis Modules.
Implementation of Real-time Anomaly Detection Algorithms.
Integration of MesoNet for Deep Fake Identification.
Development of Features for Law Enforcement Understanding.
Continuous Improvement Mechanisms for Algorithm Refinement.
User Interface Enhancement for Result Presentation.

### Code

**Model Training**

```
# Train the model with early stopping
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=16, validation_split=0.2, callbacks=[early_stopping])
```
![Screenshot (7)](https://github.com/12kaddy/RJPOLICE_HACK_569_Mobcipher_Problem-Statement-8/assets/86558772/46ecbf1c-cb49-4dc6-9ecd-f5a1d24e41b6)

**Model Testing**

![Screenshot (8)](https://github.com/12kaddy/RJPOLICE_HACK_569_Mobcipher_Problem-Statement-8/assets/86558772/38345057-e68c-4a16-bc7e-c9962ff2d18b)


**Back End**

```
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

```
### Output
![Photo from Karthik😜](https://github.com/12kaddy/RJPOLICE_HACK_569_Mobcipher_Problem-Statement-8/assets/86558772/33315b87-c99f-4487-8a9f-5a61bd5e6ebe)

![Screenshot (6)](https://github.com/12kaddy/RJPOLICE_HACK_569_Mobcipher_Problem-Statement-8/assets/86558772/b3f5fac0-32cd-4988-8ef6-e2280cc97f22)


### Timeline:
December 16, 2023 - January 10, 2024

### Notes:

Regular updates will be provided in the branch and during team meetings.
Testing should be conducted incrementally after the implementation of each key task.
Collaborators are encouraged to seek feedback and discuss challenges promptly.

