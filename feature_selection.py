from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# Separate features and labels
X = df.iloc[:, :-1]  # All 31 features
y = df["genre"]       # Genre labels

# Select top 20 features
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = df.columns[:-1][selector.get_support()]
print("Selected Features:", selected_features)

# Create new DataFrame with selected features
df_selected = pd.DataFrame(X_selected, columns=selected_features)
df_selected["genre"] = y  # Add genre column back

# Save the optimized dataset
df_selected.to_csv("optimized_data.csv", index=False)
print("Optimized dataset saved as 'optimized_data.csv'")
