import os
import librosa
import numpy as np
import pandas as pd

GENRE_PATH = "genres"
OUTPUT_CSV = "data.csv"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        return np.hstack([mfcc, chroma, spectral_contrast])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

data = []
genres = []

for genre in os.listdir(GENRE_PATH):
    genre_path = os.path.join(GENRE_PATH, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            features = extract_features(file_path)
            if features is not None:
                data.append(features)
                genres.append(genre)

df = pd.DataFrame(data)
df["genre"] = genres
df.to_csv(OUTPUT_CSV, index=False)
print("Feature extraction completed and saved to", OUTPUT_CSV)
