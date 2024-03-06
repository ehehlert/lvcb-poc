import pandas as pd
import numpy as np
from joblib import load
from scipy.sparse import hstack, csr_matrix
import fitz  # PyMuPDF

# Configure output parameters
bounding_box = True

data_filename = 'test-me.pkl'

model_filepath = 'models/cell-classification'
pdf_filepath = 'models/cell-classification/input/ready_to_draw'

# Load the trained model and components
model = load(f'{model_filepath}/lib/model.joblib')
scaler = load(f'{model_filepath}/lib/scaler.joblib')
pca = load(f'{model_filepath}/lib/pca_numerical.joblib')

# Load the vectorizers
tfidf_vectorizer_content = load('corpuses/tfidf_vectorizer_content.joblib')
tfidf_vectorizer_titles = load('corpuses/tfidf_vectorizer_titles.joblib')

# Load the new data
new_data = pd.read_pickle(f'{model_filepath}/input/ready_to_predict/{data_filename}')

new_df = pd.DataFrame(new_data)

# Prepare numerical features and scale them
feature_columns = ['cell_width', 'cell_height', 'cell_left', 'cell_top', 'row_index', 'column_index', 'row_span', 'column_span', 'table_rows', 'table_columns', 'table_cells', 'cell_type', 'entity_type'] + [col for col in new_df.columns if col.startswith('table_type_')]
tfidf_feature_columns = [col for col in new_df.columns if col.startswith('cell_content_tfidf_')]

X_numerical_new = new_df[feature_columns]
X_numerical_scaled_new = scaler.transform(X_numerical_new)

# Apply PCA to numerical features
X_numerical_pca_new = pca.transform(X_numerical_scaled_new)

# Prepare text features (pre-vectorized)
X_tfidf_cell_content_new = new_df.loc[:, new_df.columns.str.startswith('cell_content_tfidf_')]

# Convert numerical PCA features to sparse matrix
X_numerical_pca_new_sparse = csr_matrix(X_numerical_pca_new)

# Stack features horizontally
X_combined_new = hstack([X_numerical_pca_new_sparse, X_tfidf_cell_content_new])

# Make predictions
predictions = model.predict(X_combined_new)

# Create a new DataFrame with cell_id and predictions
output_df = pd.DataFrame({
    'cell_id': new_df['cell_id'],
    'cell_content': new_df['cell_content'],
    'cell_role': predictions,
    'source': new_df['source'],
    'table_page': new_df['table_page'],
    'cell_left': new_df['cell_left'],
    'cell_top': new_df['cell_top'],
    'cell_width': new_df['cell_width'],
    'cell_height': new_df['cell_height']
})

# # Define a mapping from table_class and cell_role to colors (RGB tuples)

table_color_map = {
    "BIGRID_STACKED": (1, 0, 0),  # Red
    "BIGRID": (0, 1, 0),         # Green
    "KVP": (1, 0, 1),            # Magenta
}

cell_role_map = {
    "title": (1, 0, 0),          # Red
    "key": (0, 0.5, 0),          # Dark Green
    "value": (0, 0, 1),          # Blue
    "x1": (1, 0.5, 0),           # Orange
    "x2": (0.5, 0, 0.5),         # Purple
    "y1": (1, 1, 0),             # Yellow
    "y2": (0, 0.75, 0.75),       # Dark Cyan
    "z1": (0.75, 0.5, 0.25),     # Brown
    "empty": (0.75, 0.75, 0.75), # Light Gray
}


if bounding_box:

    # Sort the DataFrame by filename and page number
    output_df.sort_values(by=['source', 'table_page'], inplace=True)

    current_pdf = None
    doc = None

    for index, row in output_df.iterrows():
        source = row['source']
        filename = f"{pdf_filepath}/{source[:-5]}.pdf"  # Adjust as necessary
        page_number = int(row['table_page']) - 1

        # Open the PDF if it's not already open
        if filename != current_pdf:
            if doc is not None:
                # Save and close the previous document
                output_filename = f"T_annotated_{current_pdf.split('/')[-1]}"
                doc.save(output_filename)
                doc.close()
                print(f"Annotated PDF saved as {output_filename}.")
            doc = fitz.open(filename)
            current_pdf = filename

        if page_number < 0 or page_number >= doc.page_count:
            continue

        page = doc.load_page(page_number)
        page_width = page.rect.width
        page_height = page.rect.height

        # Check if the cell's role is in the map before drawing
        cell_role = row['cell_role']
        if cell_role in cell_role_map:
            # Draw cell bounding box
            cell_left = row['cell_left'] * page_width
            cell_top = row['cell_top'] * page_height
            cell_width = row['cell_width'] * page_width
            cell_height = row['cell_height'] * page_height
            bbox = fitz.Rect(cell_left, cell_top, cell_left + cell_width, cell_top + cell_height)
            cell_color = cell_role_map[cell_role]
            page.draw_rect(bbox, color=cell_color, width=1)  # Adjust width as needed
            print(f"Processed cell {row['cell_id']} with role {cell_role}.")
        else:
            # Skip drawing if the cell's role is not in the map
            print(f"Skipped drawing cell {row['cell_id']} with unspecified role {cell_role}.")

        print(f"Processed cell {row['cell_id']} with role {row['cell_role']}.")

    # Save and close the last document
    if doc is not None:
        output_filename = f"annotated_{data_filename}_{current_pdf.split('/')[-1]}"
        doc.save(f"{model_filepath}/output/{output_filename}")
        doc.close()
        print(f"Annotated PDF saved as {output_filename}.")


# Save the DataFrame to a CSV file
        
predictions_output_filename = f"predictions_{data_filename}"
output_df.to_csv(f'models/cell-classification/output/{predictions_output_filename}-new.csv', index=False)

print(f"Predictions saved to {predictions_output_filename}.csv.")