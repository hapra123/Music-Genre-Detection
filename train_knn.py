import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_CSV = "data.csv"
MODEL_FILE = "knn_model.pkl"

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

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Test accuracy
accuracy = knn.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")
