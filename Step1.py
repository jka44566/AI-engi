import pandas as pd

# Load the updated dataset with composite features
file_path = '/Users/6581/Downloads/composite_vegemite.csv'
df = pd.read_csv(file_path)

# Get the number of features (columns) in the dataset
num_features = df.shape[1]  # Total number of columns

print(f"Number of features in the final dataset: {num_features}")
