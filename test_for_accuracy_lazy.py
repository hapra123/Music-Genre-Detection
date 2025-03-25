import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from lazypredict.Supervised import LazyClassifier  # Import LazyPredict

DATA_CSV = "data.csv"
#MODEL_FILE = "knn_model.pkl"

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
print(models)

# From here, you can choose the best model and proceed with training it on the full training data
best_model_name = models.index[0]  # The best model according to LazyPredict
best_model = models.loc[best_model_name]