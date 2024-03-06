import pandas as pd

df1 = pd.read_csv('1_preprocessed_tables.csv')
df2 = pd.read_csv('2_preprocessed_tables.csv')
df3 = pd.read_csv('3_preprocessed_tables.csv')

concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)

concatenated_df.to_csv('concatenated_tables.csv', index=False)