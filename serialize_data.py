import errno
import time
import os
import json
import gzip
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm


README = """Module to process downloaded goodreads json files and save in dataframe format.

    Many of the JSON files are extemely large, to save RAM processing
    will be done in chunks and appended to processed output files that
    can be used for modelling.

    General Outline:
    ----------------
        1. Read JSON into df_chunks object.
        2. Iterate over df_chunks loading chunksize (100k) rows of data into memory at a time.
        3. Apply desired filtering/processing/transformations.
        4. Append chunk of processed data to .csv
        5. Save .csv data in parquet format

    Outputs:
    -------
        goodreads_interactions.snap.parquet
        titles.snap.parquet

        books_simple_features.csv - Book numeric feature data
        books_extra_features.csv  - Book string feature data
"""


def load_json_to_df(path, head = 10_000):
    """load top head lines of data from json path
    and return pandas.DataFrame
    """
    count = 0
    data = []
    with gzip.open(path) as f:
        for l in f:
            d = json.loads(l)
            count += 1
            data.append(d)

            if (head is not None) and (count > head):
                break
    return pd.DataFrame(data)


def remove_file_if_exists(filename):
    """Remove file if present otherwise pass
    """
    try: 
        os.remove(filename)
        print(f'removed {filename}')
    except OSError as e: 
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


def extract_more_book_features(df):
    """Extracts desired features from json

    adds columns:
        ('title', 'title_without_series', 'language_code', 
         'authors', 'edition_information', 'country_code'
        )
    returns:
        pandas.DataFrame
    """
    cols = [
        'book_id', 'popular_shelves', 'description', 'format', 
        'series', 'title', 'title_without_series', 'language_code', 
        'authors', 'edition_information', 'country_code',
            ]

    dict(df[cols].dtypes)

    return df[cols].astype(
        {'book_id': 'int64',
        'popular_shelves': 'object',
        'description': 'object',
        'format': 'object',
        'series': 'object',
        'title': 'object', 
        'title_without_series': 'object',
        'language_code': 'object', 
        'authors': 'object', 
        'edition_information': 'object', 
        'country_code': 'object'})


def process_simple_book_features(df):
    """Processes 'simple' (numeric) Book Features
    inputs:
        pandas.DataFrame (book features)
    returns:
        pandas.DataFrame (numeric features only)
    """
    df['is_ebook'] = np.where(df['is_ebook'] == 'true', 1, 0)
    drop_cols = ['isbn', 'asin', 'kindle_asin', 'link', 'publisher', 'isbn13',
                'publication_day', 'publication_month', 'edition_information',
                'url', 'image_url',]

    df = df.drop(drop_cols, axis='columns')

    int_cols = ['book_id', 'text_reviews_count', 'num_pages', 'publication_year', 'work_id',
                'ratings_count']
    num_cols = ['average_rating']

    for col in int_cols + num_cols:
        try:
            df[col] = pd.to_numeric(df[col])
            if col in int_cols:
                try:
                    df[col] = df[col].astype(int)
                except Exception as e:
                    pass
                    # print(f'unable to convert {col} to int error: {e}')
        except Exception as e:
            pass
            # print(f'unable to convert column: {col} error: {e}')
    df = df.select_dtypes(include=['int', 'float', 'bool'])
    return df


def process_book_json_to_csv(large_json: str, processed_csv: str, chunksize: int = 100_000):
    """Read large json file in chunks and save to csv file.
    """
    chunks = pd.read_json(large_json, lines=True, chunksize = 100_000)
    start = time.time()

    for i, df_chunk in tqdm(enumerate(chunks)):

        dffeats = extract_more_book_features(df_chunk)
        print(f'iteration {i} clean_shape: {dffeats.shape}, original: {df_chunk.shape}')

        if i == 0:
            remove_file_if_exists(processed_csv)
            dffeats.to_csv(processed_csv, 
                        header=True, index=False)

        if i > 0:
            dffeats.to_csv(processed_csv, 
                        header=False, index=False, mode='a')

    time_seconds = time.time() - start
    print(f"Finihsed processing {large_json} and saving output to {processed_csv} in {time_seconds:.1f} seconds")


def save_interactions(input_path):
    dfi = pd.read_csv(input_path)
    dfi.to_parquet(input_path.replace('csv', 'snap.parquet'))
    return


if __name__ == '__main__':

    large_json = 'data/goodreads_books.json.gz'
    processed_csv = 'data/books_extra_features.csv'
    
    # Read book features json extract features and saves
    process_book_json_to_csv(large_json, processed_csv)

    # Read processed .csv and save as parquet for faster reads
    dfout = pd.read_csv(processed_csv)
    dfout[['book_id', 'title', 'title_without_series']].to_parquet("data/title.snap.parquet")
    dfout[['book_id', 'description', 'format', 'title', 'title_without_series',
        'language_code', 'authors', 'country_code']].to_parquet(processed_csv.replace('csv', 'snap.parquet'))
    
    # Interactions
    interactions_csv = 'data/goodreads_interactions.csv'
    save_interactions(interactions_csv)
