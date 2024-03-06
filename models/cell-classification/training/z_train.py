# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

# Load the training dataset
file_path = 'models/cell-classification/training/balanced_set_tagged.pkl'  # Update this path accordingly
data = pd.read_pickle(file_path)

print(f'Dataset loaded: {data.shape[0]} rows and {data.shape[1]} columns')

# Prepare numerical features and scale them
feature_columns = ['cell_width', 'cell_height', 'cell_left', 'cell_top', 'row_index', 'column_index', 'row_span', 'column_span', 'table_rows', 'table_columns', 'table_cells', 'cell_type', 'entity_type'] + [col for col in data.columns if col.startswith('table_type_')]
tfidf_feature_columns = [col for col in data.columns if col.startswith('cell_content_tfidf_')]

numerical_and_onehot_features = data[feature_columns]
tfidf_features = data[tfidf_feature_columns]

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(numerical_and_onehot_features)

# Ensure X_numerical_scaled is a 2D matrix; reshape if it's a 1D array
if len(X_numerical_scaled.shape) == 1:
    X_numerical_scaled = X_numerical_scaled.reshape(-1, 1)

# If X_numerical_scaled is a dense array, convert it to a sparse matrix
if isinstance(X_numerical_scaled, np.ndarray):
    X_numerical_scaled = csr_matrix(X_numerical_scaled)

# Now stack them horizontally
X_combined = hstack([X_numerical_scaled, tfidf_features])

# Combine textual and numerical features
y = data['cell_role']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualization and model saving code remains unchanged
