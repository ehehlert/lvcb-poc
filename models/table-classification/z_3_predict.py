import pandas as pd
import numpy as np
from joblib import load
from scipy.sparse import hstack, csr_matrix
import fitz  # PyMuPDF

# Configure output parameters
bounding_box = True

data_filename = 'textract_results_d84321db1a0a4854fc8accc7fbb9faa470dcb7cf56ce3b111aa73b93eabd9362.pkl'

model_filepath = 'models/table-classification'
pdf_filepath = 'models/table-classification/input/ready_to_draw'

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
feature_columns = ['table_width', 'table_height', 'table_left', 'table_top', 'cell_count', 'row_count', 'column_count', 'child_count', 'merged_cell_count', 'table_title_count', 'table_footer_count'] + [col for col in new_data.columns if col.startswith('table_type_')]
X_numerical_new = new_df[feature_columns]
X_numerical_scaled_new = scaler.transform(X_numerical_new)

# Apply PCA to numerical features
X_numerical_pca_new = pca.transform(X_numerical_scaled_new)

# Prepare text features (pre-vectorized)
content_corpus_new = new_df['content'].apply(lambda x: ' '.join(map(str, x)))
titles_corpus_new = new_df['layout_title_text'].apply(lambda x: str(x))

X_tfidf_content_new = tfidf_vectorizer_content.transform(content_corpus_new)
X_tfidf_titles_new = tfidf_vectorizer_titles.transform(titles_corpus_new)

# Convert numerical PCA features to sparse matrix
X_numerical_pca_new_sparse = csr_matrix(X_numerical_pca_new)

# Stack features horizontally
X_combined_new = hstack([X_numerical_pca_new_sparse, X_tfidf_content_new, X_tfidf_titles_new])

# Make predictions
predictions = model.predict(X_combined_new)

# Create a new DataFrame with table_id, content as a single string, and predictions
output_df = pd.DataFrame({
    'table_id': new_df['table_id'],
    'table_content': new_df['content'].apply(lambda x: ' '.join(map(str, x))),
    'table_width': new_df['table_width'],
    'table_height': new_df['table_height'],
    'table_left': new_df['table_left'],
    'table_top': new_df['table_top'],
    'source': new_df['source'],
    'table_page': new_df['table_page'],
    'table_class': predictions
})

# # Define a mapping from table_class and cell_role to colors (RGB tuples)

table_color_map = {
    "BIGRID_STACKED": (1, 0, 0),  # Red
    "BIGRID": (0, 1, 0),         # Green
    "KVP": (1, 0, 1),            # Magenta
}

if bounding_box:

    # Sort the DataFrame by filename and page number
    output_df.sort_values(by=['source', 'table_page'], inplace=True)

    current_pdf = None
    doc = None

    for index, table_row in output_df.iterrows():
        table_id = table_row['table_id']
        filename = pdf_filepath + "/" + table_row['source'][:-5] + ".pdf" ## this is hacky
        table_class = table_row['table_class']

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

        table_page_index = int(table_row['table_page']) - 1

        if table_page_index < 0 or table_page_index >= doc.page_count:
            print(f"Error: Page index {table_page_index} is out of range for this document.")
            continue

        page = doc.load_page(table_page_index)
        page_width = page.rect.width
        page_height = page.rect.height

        # Draw the table bounding box
        abs_left = table_row['table_left'] * page_width
        abs_top = table_row['table_top'] * page_height
        abs_right = abs_left + (table_row['table_width'] * page_width)
        abs_bottom = abs_top + (table_row['table_height'] * page_height)
        bbox = fitz.Rect(abs_left, abs_top, abs_right, abs_bottom)
        table_color = table_color_map.get(table_class, (0.5, 0.5, 0.5))  # Default to gray
        page.draw_rect(bbox, color=table_color, width=2)

        print(f"Processed table {table_id} and its cells.")

    # Save and close the last document
    if doc is not None:
        output_filename = f"annotated_{data_filename}_{current_pdf.split('/')[-1]}"
        doc.save(f"models/table-classification/output/{output_filename}")
        doc.close()
        print(f"Annotated PDF saved as {output_filename}.")


# Save the DataFrame to a CSV file
        
output_df.drop(columns=['table_width', 'table_height', 'table_top', 'table_left'], inplace=True)

predictions_output_filename = f"predictions_{data_filename}"
output_df.to_csv(f'models/table-classification/output/{predictions_output_filename}.csv', index=False)

print(f"Predictions saved to {predictions_output_filename}.csv.")