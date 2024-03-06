First, parse the document using parser.py in the scripts folder. The document needs to be a PDF located in the specified S3 bucket. Next, use the preprocessor.ipynb to extract cells.csv and tables.csv from the JSON returned by Textract. If you need to process a huge amount of data, use cocat_dfs_util.py to concatenate the resulting cells.csv and tables.csv files.

Second, use the cluster_tables.ipynb model to KNN-cluster your tables data. A CSV will be output which you can add a table_class column to; filter the CSV in excel by cluster and add manual tags to the dataset. Save it.

Finally, use the classify_tables.ipynb model to train a supervised Random Forest classification model on your tagged dataset to predict table type.

Onward to cell extraction....
