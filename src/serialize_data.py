import datetime
import errno
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


def load_json_to_df(path, head=10_000):
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
    """Remove file if present otherwise pass"""
    try:
        os.remove(filename)
        print(f"removed {filename}")
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


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
        "book_id",
        "popular_shelves",
        "description",
        "format",
        "series",
        "title",
        "title_without_series",
        "language_code",
        "authors",
        "edition_information",
        "country_code",
    ]

    dict(df[cols].dtypes)

    return df[cols].astype(
        {
            "book_id": "int64",
            "popular_shelves": "object",
            "description": "object",
            "format": "object",
            "series": "object",
            "title": "object",
            "title_without_series": "object",
            "language_code": "object",
            "authors": "object",
            "edition_information": "object",
            "country_code": "object",
        }
    )


def process_simple_book_features(df):
    """Processes 'simple' (numeric) Book Features
    inputs:
        pandas.DataFrame (book features)
    returns:
        pandas.DataFrame (numeric features only)
    """
    df["is_ebook"] = np.where(df["is_ebook"] == "true", 1, 0)
    drop_cols = [
        "isbn",
        "asin",
        "kindle_asin",
        "link",
        "publisher",
        "isbn13",
        "publication_day",
        "publication_month",
        "edition_information",
        "url",
        "image_url",
    ]

    df = df.drop(drop_cols, axis="columns")

    int_cols = ["book_id", "text_reviews_count", "num_pages", "publication_year", "work_id", "ratings_count"]
    num_cols = ["average_rating"]

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
    df = df.select_dtypes(include=["int", "float", "bool"])
    return df


def save_book_features(large_json: str, processed_csv: str, chunksize: int = 100_000):
    """Read large json file in chunks and save to csv file."""
    chunks = pd.read_json(large_json, lines=True, chunksize=100_000)
    start = time.time()

    for i, df_chunk in tqdm(enumerate(chunks)):

        dffeats = extract_more_book_features(df_chunk)
        print(f"iteration {i} clean_shape: {dffeats.shape}, original: {df_chunk.shape}")

        if i == 0:
            remove_file_if_exists(processed_csv)
            dffeats.to_csv(processed_csv, header=True, index=False)

        if i > 0:
            dffeats.to_csv(processed_csv, header=False, index=False, mode="a")

    dfout = pd.read_csv(processed_csv)
    dfout[["book_id", "title", "title_without_series"]].to_parquet("data/titles.snap.parquet")
    column_order = [
        "book_id",
        "description",
        "format",
        "title",
        "title_without_series",
        "language_code",
        "authors",
        "country_code",
    ]
    dfout[column_order].to_parquet(processed_csv.replace("csv", "snap.parquet"))
    time_seconds = time.time() - start
    print(f"Finihsed processing {large_json} and saving output to {processed_csv} in {time_seconds:.1f} seconds")


def save_simple_book_features(large_json: str, output_file: str, chunksize: int = 100_000):
    # Process and Store simple (numeric) book features
    chunks = pd.read_json(large_json, lines=True, chunksize=100_000)
    start = time.time()

    for i, df_chunk in tqdm(enumerate(chunks)):
        dfclean = process_simple_book_features(df_chunk)
        print(f"iteration {i} clean_shape: {dfclean.shape}, original: {df_chunk.shape}")

        if i == 0:
            remove_file_if_exists(output_file)
            dfclean.to_csv(output_file, header=True, index=False)

        if i > 0:
            dfclean.to_csv(output_file, header=False, index=False, mode="a")

    dfout = pd.read_csv(output_file)
    column_order = [
        "book_id",
        "work_id",
        "publication_year",
        "is_ebook",
        "num_pages",
        "ratings_count",
        "text_reviews_count",
        "average_rating",
    ]
    dfout[column_order].to_parquet(output_file.replace("csv", "snap.parquet"))
    time_seconds = time.time() - start
    print(f"Finnished processing {large_json} and saving output to {output_file} in {time_seconds:.1f} seconds")
    return


def save_interactions(input_path):
    dfi = pd.read_csv(input_path)
    dfi.to_parquet(input_path.replace("csv", "snap.parquet"))
    return


def main(json_path, csv_path, interactions_path, output_path, chunksize=100_000):
    """Serialze DataFrames to parquet files.

    Args:
        json_path (_type_): _description_
        csv_path (_type_): _description_
        interactions_path (_type_): _description_
        output_path (_type_): _description_
        chunksize (_type_, optional): _description_. Defaults to 100_000.
    """
    save_book_features(json_path, csv_path, chunksize=chunksize)
    save_simple_book_features(json_path, output_path, chunksize=chunksize)
    save_interactions(interactions_path)
    return


if __name__ == "__main__":

    CHUNK_SIZE = 100_000
    LARGE_JSON = Path("data").joinpath("goodreads_books.json.gz")

    # Read book features and save to csv/parquet
    CSV_PATH = Path("data").joinpath("books_extra_features.csv")
    OUTPUT_PATH = Path("data").joinpath("books_simple_features.csv")
    INTERACTIONS_PATH = Path("data").joinpath("interactions.csv")
    main(json_path=LARGE_JSON, csv_path=CSV_PATH, interactions_path=INTERACTIONS_PATH, output_path=OUTPUT_PATH, chunksize=CHUNK_SIZE)
