import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the saved Random Forest model
model_path = '/Users/6581/Downloads/forestmodel.pkl'  # Path to your saved model
model = joblib.load(model_path)

# Step 2: Load the 1000 rows dataset
test_data_path = '/Users/6581/Downloads/sampled_vegemite.csv'  # Path to your 1000 rows dataset
df_test = pd.read_csv(test_data_path)

# New features to use
selected_features = [
    'TFE Vacuum pressure PV', 
    'TFE Steam temperature', 
    'TFE Steam pressure PV', 
    'TFE Production solids density', 
    'TFE Feed pump'
]

# Step 3: Prepare the data
# Check if the selected features are in the dataframe
for feature in selected_features:
    if feature not in df_test.columns:
        raise ValueError(f"Feature '{feature}' is not in the dataset columns")

# Extract the relevant features
X_test_new = df_test[selected_features]

# Check if scaler exists, otherwise create and fit a new one
scaler_path = '/Users/6581/Downloads/scaler.pkl'  # Path to your saved scaler
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    # If scaler not found, fit a new scaler on the training data
    scaler = StandardScaler()
    X_test_new_scaled = scaler.fit_transform(X_test_new)
    joblib.dump(scaler, scaler_path)
else:
    # Use the existing scaler to transform the test data
    X_test_new_scaled = scaler.transform(X_test_new)

# Step 4: Make predictions
y_pred_new = model.predict(X_test_new_scaled)

# Step 5: Compare predictions with the original labels
# Assuming 'Class' column contains the true labels
y_true = df_test['Class']

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred_new))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_new))

# Optional: Add predictions to the dataframe and save
df_test['Predicted_Class'] = y_pred_new
df_test.to_csv('/Users/6581/Downloads/predictions.csv', index=False)
