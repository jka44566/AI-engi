import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/6581/Downloads/composite_vegemite.csv'
df = pd.read_csv(file_path)

# Select relevant features
selected_features = ['HeatTemp_Sum', 'TFE Tank level', 'TFE Steam temperature', 'TFE Steam pressure PV', 'SteamPressure_TemperatureRatio']
X = df[selected_features]
y = df['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Store metrics
results = []

# Evaluate models
for model_name, model in models.items():
    # Train model
    model.fit(X_train_scaled if model_name != 'Random Forest' else X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled if model_name != 'Random Forest' else X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': report['1']['precision'],  # Assuming '1' is the positive class
        'Recall': report['1']['recall'],
        'F1 Score': report['1']['f1-score']
    }
    results.append(metrics)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame(results)

print(comparison_df)
