import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib

# Load your dataset
df = pd.read_csv('/Users/6581/Downloads/predictions_with_labels.csv')  # Update with the correct path

# Step 1: Select columns ending with 'SP'
sp_features = [col for col in df.columns if col.endswith('SP')]
df_sp = df[sp_features + ['Class']]  # Include 'Class' column for modeling

# Step 2: Prepare data
X = df_sp[sp_features]
y = df_sp['Class']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Print the decision tree rules
tree_rules = export_text(clf, feature_names=list(sp_features))
print("Decision Tree Rules:\n")
print(tree_rules)

# Optional: Save the trained model (if needed)
model_path = '/Users/6581/Downloads/decision_tree_model.pkl'  # Update with the correct path
joblib.dump(clf, model_path)
print(f"Decision Tree model successfully saved to {model_path}")

# Optional: Load the saved model (for future use)
# clf_loaded = joblib.load(model_path)
