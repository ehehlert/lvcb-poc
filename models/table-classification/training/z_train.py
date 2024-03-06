# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load the training dataset
file_path = 'models/table-classification/training/balanced_set_tagged.pkl'  # Update this path
data = pd.read_pickle(file_path)

print(f'Dataset loaded: {data.shape[0]} rows and {data.shape[1]} columns')

# Prepare numerical features and scale them
feature_columns = ['table_width', 'table_height', 'table_left', 'table_top', 'cell_count', 'row_count', 'column_count', 'child_count', 'merged_cell_count', 'table_title_count', 'table_footer_count'] + [col for col in data.columns if col.startswith('table_type_')]
numerical_and_onehot_features = data[feature_columns]

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(numerical_and_onehot_features)

pca_numerical = PCA(n_components=0.95)
X_numerical_pca = pca_numerical.fit_transform(X_numerical_scaled)

# Prepare text features (pre-vectorized)
X_tfidf_features = data.loc[:, data.columns.str.startswith('title_tfidf_') | data.columns.str.startswith('content_tfidf_')]

# REMOVING THESE FROM THE TRAINING SET UNTIL FURTHER NOTICE - THEY'RE NOT BEING USED
# Prepare label encoded features
# X_label_encoded = data.loc[:, data.columns.str.startswith('entity_feature_')]

from scipy.sparse import hstack, csr_matrix
import numpy as np

# Ensure X_numerical_scaled is a 2D matrix; reshape if it's a 1D array
if len(X_numerical_pca.shape) == 1:
    X_numerical_pca = X_numerical_pca.reshape(-1, 1)

# If X_numerical_scaled is a dense array, convert it to a sparse matrix
if isinstance(X_numerical_pca, np.ndarray):
    X_numerical_pca = csr_matrix(X_numerical_pca)

# Now stack them horizontally
X_combined = hstack([X_numerical_pca, X_tfidf_features])

# Combine textual and numerical features
y = data['table_class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, random_state=42)

nan_columns = data.isna().any()
columns_with_nan = nan_columns[nan_columns].index.tolist()
print("Columns with NaN:", columns_with_nan)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Get precision, recall, and F1 scores for each class
scores = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_pred))
labels = np.unique(y_pred)

# Visualization
fig, ax = plt.subplots(3, 1, figsize=(5, 5))

# Precision, Recall, and F1 Scores by Class
metrics = ['Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics):
    ax[i].barh(labels, scores[i])
    ax[i].set_title(f'{metric} by Class')
    ax[i].set_xlabel(metric)
    ax[i].set_ylabel('Class')
plt.tight_layout()
# plt.show()

from joblib import dump

# Save the trained model to a file
dump(model, 'table-classification-model-trained.joblib')
dump(scaler, 'table-classification-scaler.joblib')
dump(pca_numerical, 'table-classification-pca_numerical.joblib')
