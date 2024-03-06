import fitz  # PyMuPDF
import pandas as pd

# Load the table data and cell geometry data
tables_df = pd.read_csv('A_tables.csv')
cells_df = pd.read_csv('A_cells_geo_tagged.csv')

# Define a mapping from table_class and cell_role to colors (RGB tuples)
table_color_map = {
    "BIGRID_STACKED": (1, 0, 0),  # Red
    "BIGRID": (0, 1, 0),         # Green
    "KVP": (1, 0, 1),            # Magenta
    # Add more mappings as necessary for table_class
}

cell_color_map = {
    "comment": (0, 0, 1),        # Blue
    "empty": (1, 1, 0),          # Yellow
    "key": (0, 1, 1),         # Cyan
    "value": (1, 0.5, 0),        # Orange
    "key_1_x": (0.5, 0, 0),      # Dark red
    "key_2_x": (0, 0.5, 0),      # Dark green
    "key_1_y": (0.5, 0, 0.5),    # Dark magenta
    "key_2_y": (0, 0.5, 0.5),    # Dark cyan
    "key_diagonal": (0.5, 0.5, 0),  # Dark yellow
    "merged_helper": (0.5, 0.5, 0.5),  # Dark gray
    "title": (0.75, 0.75, 0.75),  # Light gray
    # Add more mappings as necessary for cell_role
}

# Sort the DataFrame by filename and page number
tables_df.sort_values(by=['filename', 'table_page'], inplace=True)

current_pdf = None
doc = None

for index, table_row in tables_df.iterrows():
    table_id = table_row['table_id']
    filename = table_row['filename']
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
    #abs_left = table_row['table_left'] * page_width
    #abs_top = table_row['table_top'] * page_height
    #abs_right = abs_left + (table_row['table_width'] * page_width)
    #abs_bottom = abs_top + (table_row['table_height'] * page_height)
    #bbox = fitz.Rect(abs_left, abs_top, abs_right, abs_bottom)
    #table_color = table_color_map.get(table_class, (0.5, 0.5, 0.5))  # Default to gray
    #page.draw_rect(bbox, color=table_color, width=2)

    # Process cells for the current table
    cells_for_table = cells_df[cells_df['table_id'] == table_id]
    for _, cell_row in cells_for_table.iterrows():
        cell_role = cell_row['cell_role']
        abs_left = cell_row['left'] * page_width
        abs_top = cell_row['top'] * page_height
        abs_right = abs_left + (cell_row['width'] * page_width)
        abs_bottom = abs_top + (cell_row['height'] * page_height)
        cell_bbox = fitz.Rect(abs_left, abs_top, abs_right, abs_bottom)
        cell_color = cell_color_map.get(cell_role, (0.75, 0.75, 0.75))  # Default to light gray
        page.draw_rect(cell_bbox, color=cell_color, width=1)  # Use a slightly thinner line for cells

    print(f"Processed table {table_id} and its cells.")

# Save and close the last document
if doc is not None:
    output_filename = f"T_annotated_{current_pdf.split('/')[-1]}"
    doc.save(output_filename)
    doc.close()
    print(f"Annotated PDF saved as {output_filename}.")
