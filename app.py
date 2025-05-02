import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from pydub import AudioSegment
import logging
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)
app.logger.info("Flask app started")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best_english_deep_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Helper Functions
def convert_to_wav(in_path: str, out_path: str) -> str:
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(out_path, format="wav")
    return out_path

def remove_noise(y: np.ndarray, sr: int) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    return y_trimmed

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfccs, chroma, contrast, zcr, rmse])

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Save and preprocess audio
        in_path = "input_audio"
        out_path = "processed_audio.wav"
        file.save(in_path)
        convert_to_wav(in_path, out_path)
        y, sr = librosa.load(out_path, sr=16000)
        y = remove_noise(y, sr)
        features = extract_features(y, sr)
        features_scaled = scaler.transform([features])

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], features_scaled.astype(np.float32))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Prediction
        predicted_class = np.argmax(output_data)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

        # Cleanup
        os.remove(in_path)
        os.remove(out_path)

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Health Check Endpoint
@app.route('/', methods=['GET'])
def index():
    return "Flask API is running successfully🚀"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
