import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Import, process, and combine the data from multiple JSON files

shared_path = './models/table-classification/input/ready_to_preprocess'

json_files = [f'{shared_path}/balanced_set.json']

output_name = 'balanced_set'

def process_data(data, source_file_name):

    id_to_item = {item['Id']: item for item in data}
    
    parent_to_all_children = {}
    for item in data:
        if 'Relationships' in item:
            for relationship in item['Relationships']:
                # Initialize a dictionary for the item if it doesn't exist
                if item['Id'] not in parent_to_all_children:
                    parent_to_all_children[item['Id']] = {}
                # Append the child IDs under the appropriate relationship type
                parent_to_all_children[item['Id']].setdefault(relationship['Type'], []).extend(relationship['Ids'])
    
    cell_records = []

    for item in data:
        if item.get('BlockType') == 'TABLE':
            table_id = item['Id']
            relationships = parent_to_all_children.get(table_id, {})
            
            # Initialize an empty list to hold cell records, including cell_type
            aggregated_cells = []
            for rel_type in ['CHILD', 'MERGED_CELL', 'TABLE_FOOTER', 'TABLE_TITLE']:
                cell_ids = relationships.get(rel_type, [])
                for cell_id in cell_ids:
                    # Append both cell_id and its relationship type to the list
                    aggregated_cells.append((cell_id, rel_type))
            
            for cell_id, cell_type in aggregated_cells:
                cell_block = id_to_item.get(cell_id)
                if not cell_block:
                    continue

                entity_type = cell_block.get('EntityTypes', [None])[0] if 'EntityTypes' in cell_block else None

                cell_geometry = cell_block['Geometry']['BoundingBox']
                
                child_ids = parent_to_all_children.get(cell_id, {}).get('CHILD', [])
                cell_words = [id_to_item[child_id]['Text'] for child_id in child_ids if child_id in id_to_item and 'Text' in id_to_item[child_id]]
                cell_content = ' '.join(cell_words)
                
                cell_records.append({
                    'cell_id': cell_id,
                    'cell_type': cell_type,  # Include the cell_type here
                    'entity_type': entity_type,
                    'cell_words': cell_words,
                    'cell_content': cell_content,
                    'cell_width': cell_geometry['Width'],
                    'cell_height': cell_geometry['Height'],
                    'cell_left': cell_geometry['Left'],
                    'cell_top': cell_geometry['Top'],
                    'row_index': cell_block.get('RowIndex', None),
                    'column_index': cell_block.get('ColumnIndex', None),
                    'row_span': cell_block.get('RowSpan', 1),
                    'column_span': cell_block.get('ColumnSpan', 1),
                    'table_id': table_id,
                    'table_type': item['EntityTypes'][0] if 'EntityTypes' in item else None,
                    'table_width': item['Geometry']['BoundingBox']['Width'],
                    'table_height': item['Geometry']['BoundingBox']['Height'],
                    'table_left': item['Geometry']['BoundingBox']['Left'],
                    'table_top': item['Geometry']['BoundingBox']['Top'],
                    'table_page': item['Page'],
                    'source': source_file_name,
                    # Additional fields can be added here
                })
    return pd.DataFrame(cell_records)

# Process each file and combine the results as before
cells_dfs = [process_data(data, file_path.split('/')[-1]) for file_path in json_files for data in [json.load(open(file_path))]]
cells_dfs = pd.concat(cells_dfs, ignore_index=True)

# Display the length of the dataframe
print(f'Combined Cells Dataframe has {cells_dfs.shape[0]} rows and {cells_dfs.shape[1]} columns')

# Establish meaningful cell content for each cell by examining merged cell relationships


# Ensure the necessary columns are in the correct data type
cells_dfs['row_index'] = cells_dfs['row_index'].fillna(0).astype(int)
cells_dfs['column_index'] = cells_dfs['column_index'].fillna(0).astype(int)
cells_dfs['row_span'] = cells_dfs['row_span'].fillna(1).astype(int)  # Default span of 1 if missing
cells_dfs['column_span'] = cells_dfs['column_span'].fillna(1).astype(int)

# Sort the DataFrame as required
cells_dfs.sort_values(by=['table_id', 'column_index', 'row_index'], inplace=True)

# Isolate merged cells
merged_cells = cells_dfs[cells_dfs['cell_type'] == 'MERGED_CELL']

# Initialize a column for tracking merged cell parent ID and merge status
cells_dfs['merged_parent_cell_id'] = np.nan
cells_dfs['has_merged_parent'] = 0

for cell in merged_cells.itertuples():
    # Calculate the affected range of rows and columns
    affected_rows = range(cell.row_index, cell.row_index + cell.row_span)
    affected_columns = range(cell.column_index, cell.column_index + cell.column_span)

    # Find the cells that are affected
    affected_cells = cells_dfs[
        (cells_dfs['table_id'] == cell.table_id) &
        (cells_dfs['cell_type'] == 'CHILD') &  # Targeting only child cells
        (cells_dfs['row_index'].isin(affected_rows)) &
        (cells_dfs['column_index'].isin(affected_columns))
    ]

    # Aggregate text content of affected cells, stripping to remove leading/trailing spaces
    aggregated_text_content = " ".join(filter(None, affected_cells['cell_content'].astype(str))).strip()

    if aggregated_text_content:
        # Update the affected cells with the aggregated text content and merge-related information
        cells_dfs.loc[affected_cells.index, 'cell_content'] = aggregated_text_content
        cells_dfs.loc[affected_cells.index, 'merged_parent_cell_id'] = cell.cell_id
        cells_dfs.loc[affected_cells.index, 'has_merged_parent'] = 1

# Fill missing values for new columns
cells_dfs['has_merged_parent'] = cells_dfs['has_merged_parent'].fillna(0).astype(int)
# Do not convert merged_parent_cell_id to int; leave it as is or ensure it's treated as a string/object
cells_dfs['merged_parent_cell_id'] = cells_dfs['merged_parent_cell_id'].fillna('None')

# Optional: If you want to ensure 'merged_parent_cell_id' is explicitly recognized as a string/object column:
cells_dfs['merged_parent_cell_id'] = cells_dfs['merged_parent_cell_id'].astype(str)

# Display the length of the DataFrame
print(f'Processed Cells DataFrame has {cells_dfs.shape[0]} rows and {cells_dfs.shape[1]} columns')

# Summarize cell contents and child entities for each table in tables_df_without_titles


# Function to aggregate cell contents into a list of lists, one per row
def aggregate_contents(group):
    # Sort the group by row and column index to ensure the correct order
    sorted_group = group.sort_values(by=['row_index', 'column_index'])
    # Aggregate contents by row
    contents_by_row = sorted_group.groupby('row_index')['cell_words'].apply(list).tolist()
    return contents_by_row

def aggregate_child_entities(group):
    # Filter the group to only include CHILD cells
    child_cells = group[group['cell_type'] == 'CHILD']
    # Replace NaN or empty entity_type values with 'normal'
    child_cells['entity_type'] = child_cells['entity_type'].replace({np.nan: 'normal', '': 'normal'})
    # Sort the group by row and column index to ensure the correct order
    sorted_group = child_cells.sort_values(by=['row_index', 'column_index'])
    # Aggregate entity types by row
    entities_by_row = sorted_group.groupby('row_index')['entity_type'].apply(list).tolist()
    return entities_by_row

# Aggregate information for each table
tables_df_without_titles = cells_dfs.groupby('table_id').apply(lambda g: pd.Series({
    'table_width': g['table_width'].max(),
    'table_height': g['table_height'].max(),
    'table_left': g['table_left'].max(),
    'table_top': g['table_top'].max(),
    'table_page': g['table_page'].max(),
    'source': g['source'].iloc[0],
    'cell_count': g['cell_words'].count(),
    'row_count': int(g['row_index'].max()),
    'column_count': int(g['column_index'].max()),
    'content': aggregate_contents(g),
    'entities': aggregate_child_entities(g),
    # Add counts for different cell types
    'child_count': g[g['cell_type'] == 'CHILD']['cell_type'].count(),
    'merged_cell_count': g[g['cell_type'] == 'MERGED_CELL']['cell_type'].count(),
    'table_title_count': g[g['cell_type'] == 'TABLE_TITLE']['cell_type'].count(),
    'table_footer_count': g[g['cell_type'] == 'TABLE_FOOTER']['cell_type'].count(),
    'table_type': g['table_type'].max()
})).reset_index()


# Calculate page title per page of each document based on confidence scores, store in titles_df, and merge with tables_df_without_titles as tables_df


def process_layout_titles(data, source_file_name):
    layout_title_ids = [item['Id'] for item in data if item.get('BlockType') == 'LAYOUT_TITLE']
    id_to_item = {item['Id']: item for item in data}
    
    layout_titles = []
    for layout_title in layout_title_ids:
        layout_title_block = id_to_item[layout_title]
        layout_title_cell = {
            'layout_title_id': layout_title,
            'layout_title_text': ' '.join([id_to_item[child_id]['Text'] for child_id in layout_title_block.get('Relationships', [{}])[0].get('Ids', []) if child_id in id_to_item and 'Text' in id_to_item[child_id]]),
            'layout_title_page': layout_title_block['Page'],
            'layout_title_confidence': layout_title_block['Confidence'],
            'source': source_file_name,  # Keep track of the source document
        }
        layout_titles.append(layout_title_cell)

    doc_titles_df = pd.DataFrame(layout_titles)
    
    # Perform calculations within the current document's scope
    doc_titles_df['max_confidence_per_page'] = doc_titles_df.groupby('layout_title_page')['layout_title_confidence'].transform('max')
    doc_titles_df['is_max_confidence'] = doc_titles_df['layout_title_confidence'] == doc_titles_df['max_confidence_per_page']
    doc_titles_df.drop(columns=['max_confidence_per_page'], inplace=True)
    
    return doc_titles_df

# Initialize an empty list to hold DataFrames from all files
titles_df = []

# Process each JSON file separately and append the results to the list
for file_path in json_files:
    with open(file_path) as file:
        data = json.load(file)
        doc_titles_df = process_layout_titles(data, file_path.split('/')[-1])
        titles_df.append(doc_titles_df)

# Concatenate all DataFrames after processing
titles_df = pd.concat(titles_df, ignore_index=True)

tables_df = pd.merge(tables_df_without_titles, titles_df[titles_df['is_max_confidence'] == True][['source', 'layout_title_page', 'layout_title_text']],
                     left_on=['source', 'table_page'], right_on=['source', 'layout_title_page'], how='left')

tables_df.drop(columns=['layout_title_page'], inplace=True, errors='ignore')

print(f'Combined Tables DataFrame has {tables_df.shape[0]} rows and {tables_df.shape[1]} columns')

# Preprocessing steps for the table content and entities


## ONE-HOT ENCODING FOR TABLE_TYPE
one_hot_encoder = OneHotEncoder()

table_type_encoded = one_hot_encoder.fit_transform(tables_df[['table_type']])
table_type_encoded_dense = table_type_encoded.toarray()
column_names = one_hot_encoder.get_feature_names_out(['table_type'])
table_type_encoded_df = pd.DataFrame(table_type_encoded_dense, columns=column_names)

tables_df = pd.concat([tables_df.reset_index(drop=True), table_type_encoded_df.reset_index(drop=True)], axis=1)

print(f'Completed one-hot encoding for table_type. New shape: {tables_df.shape}')

from joblib import load

tfidf_vectorizer_content = load('./corpuses//tfidf_vectorizer_content.joblib')
tfidf_vectorizer_titles = load('./corpuses/tfidf_vectorizer_titles.joblib')

new_content_corpus = tables_df['content'].apply(lambda x: ' '.join(map(str, x)))
new_titles_corpus = tables_df['layout_title_text'].apply(lambda x: str(x))  # Assuming this is already a string or similar operation if needed

tfidf_layout_title_features = tfidf_vectorizer_titles.transform(new_titles_corpus)
tfidf_content_features = tfidf_vectorizer_content.transform(new_content_corpus)

# For content features
content_feature_names = [f'content_tfidf_{i}' for i in range(tfidf_content_features.shape[1])]
tfidf_content_df = pd.DataFrame(tfidf_content_features.toarray(), columns=content_feature_names)

# For title features
title_feature_names = [f'title_tfidf_{i}' for i in range(tfidf_layout_title_features.shape[1])]
tfidf_layout_title_df = pd.DataFrame(tfidf_layout_title_features.toarray(), columns=title_feature_names)

# Reset index on the original DataFrame if necessary to ensure alignment
tables_df.reset_index(drop=True, inplace=True)

# Concatenate the original DataFrame with the new TF-IDF DataFrames
tables_df = pd.concat([tables_df, tfidf_content_df, tfidf_layout_title_df], axis=1)

print('Completed TF-IDF vectorization for content and titles:', tables_df.shape)

tables_df.to_pickle(f'./models/table-classification/input/ready_to_predict/{output_name}.pkl')
# tables_df.to_csv(f'./models/table-classification/input/ready_to_predict/{output_name}.csv', index=False)

print(f'Preprocessed df saved to models/table-classification/input/ready_to_predict/{output_name}.pkl')