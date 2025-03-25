import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
from lazypredict.Supervised import LazyClassifier

DATA_CSV = "data.csv"
MODEL_FILE = "calibrated_svc_model.pkl"

# Load dataset
df = pd.read_csv(DATA_CSV)
X = df.iloc[:, :-1].values
y = df["genre"].values

# Encode genre labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LazyClassifier and fit on the dataset
clf = LazyClassifier()
models, _ = clf.fit(X_train, X_test, y_train, y_test)

# Show results of all models

# Use Support Vector Classifier (SVC) as the base model
svc = SVC(probability=True, random_state=42)  # Ensure probability=True to use CalibratedClassifierCV

# Apply CalibratedClassifierCV
calibrated_svc = CalibratedClassifierCV(svc, method='sigmoid', cv='prefit')

# Fit the base model first
svc.fit(X_train, y_train)

# Calibrate the classifier
calibrated_svc.fit(X_train, y_train)

# Save the calibrated model and other components
joblib.dump(calibrated_svc, MODEL_FILE)
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Test accuracy
accuracy = calibrated_svc.score(X_test, y_test)
print(f"Calibrated SVC model trained with accuracy: {accuracy:.2f}")
