import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load your dataset
df = pd.read_csv('/Users/6581/Downloads/composite_vegemite.csv')

# Define your features and target
selected_features = ['HeatTemp_Sum', 'TFE Tank level', 'TFE Steam temperature', 'TFE Steam pressure PV', 'SteamPressure_TemperatureRatio']
X = df[selected_features]
y = df['Class']  # Replace 'Class' with the actual name of your target column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
model_path = '/Users/6581/Downloads/forestmodel.pkl'  # Update this path as needed
joblib.dump(model, model_path)
print(f"Model successfully saved to {model_path}")

# Optional: Print feature importances
importances = model.feature_importances_
print("Feature Importances:")
for feature, importance in zip(selected_features, importances):
    print(f"{feature}: {importance:.4f}")
