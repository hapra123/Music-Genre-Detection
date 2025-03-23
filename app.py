from flask import Flask, request, render_template
import os
import numpy as np
import librosa
import joblib

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Allow 50MB uploads

# Set Upload Directory (Temporary storage for Render deployment)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained models and scaler
knn = joblib.load("knn_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Function to extract audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        features = np.hstack([mfcc, chroma, spectral_contrast])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_audio.wav")
        file.save(file_path)

        features = extract_features(file_path)
        if features is not None:
            features = scaler.transform([features])

            # Predict the genre
            genre_index = knn.predict(features)[0]
            genre = encoder.inverse_transform([genre_index])[0]

            # Get confidence score if available
            try:
                probabilities = knn.predict_proba(features)[0]
                confidence = np.max(probabilities) * 100  # Convert to percentage
            except AttributeError:
                confidence = 100  # Some models don't support `predict_proba`

            # If confidence is low, classify as "Unknown Genre"
            if confidence < 60:
                genre = "Unknown Genre"

            return render_template("index.html", genre=genre, confidence=round(confidence, 2))

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(debug=False, host="0.0.0.0", port=port)
