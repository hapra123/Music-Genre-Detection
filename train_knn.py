import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

DATA_CSV = "data.csv"
#MODEL_FILE = "knn_model.pkl"

# Load the dataset
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

# Define the range of k values to test
k_values = range(1, 21)  # Testing k from 1 to 20
best_k = 1
best_accuracy = 0

# Perform K-Fold Cross-Validation to find the best k value
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy')  # Cross-validation for each k
    avg_accuracy = np.mean(cv_scores)  # Average accuracy across folds
    
    print(f"K={k}, Cross-Validation Accuracy: {avg_accuracy:.2f}")
    
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_k = k

print(f"Optimal value of k: {best_k} with an average cross-validation accuracy of {best_accuracy:.2f}")

# Train the model with the best k
knn_optimal = KNeighborsClassifier(n_neighbors=best_k)
knn_optimal.fit(X_train, y_train)

# Test accuracy
accuracy = knn_optimal.score(X_test, y_test)
print(f"Model trained with optimal k={best_k} and accuracy on test set: {accuracy:.2f}")
