from flask import Flask, request, render_template
import numpy as np
import librosa
import joblib

app = Flask(__name__)

# Load trained models and scaler
knn = joblib.load("calibrated_svc_model.pkl")
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
        file = request.files["file"]
        if file:
            file_path = "uploaded_audio.wav"
            file.save(file_path)

            features = extract_features(file_path)
            if features is not None:
                features = scaler.transform([features])
                
                # Predict the genre
                genre_index = knn.predict(features)[0]
                genre = encoder.inverse_transform([genre_index])[0]
                
                # Get confidence score
                probabilities = knn.predict_proba(features)[0]
                confidence = np.max(probabilities) * 100  # Convert to percentage
                
                # If confidence is low, classify as "Unknown Genre"
                if confidence < 60:
                    genre = "Unknown Genre"

                return render_template("index.html", genre=genre, confidence=round(confidence, 2))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
