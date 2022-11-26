import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# PARAMS
TOP_BOOKS = None
TOP_USERS = None


def create_matrix(dfi: pd.DataFrame, dfbooks: pd.DataFrame, top_books=None, top_users=None) -> pd.DataFrame:
    """Create a matrix of users and books with the number of reads as the value."""
    start = time.time()
    logger.info("Begin Creating Feature Matrix")
    dff = dfi.set_index("book_id").join(dfbooks[["book_id", "title"]].set_index("book_id"), how="inner").reset_index()

    topu = dff.groupby("user_id")["book_id"].nunique()
    topu = topu.sort_values(ascending=False).head(top_users)

    topb = dff.groupby("book_id").agg({"user_id": "nunique", "is_reviewed": "sum", "is_read": "sum", "rating": "mean"})
    # bookrate = dff[dff["is_read"] == 1].groupby("book_id")["rating"].mean()
    # topb = topb.join(bookrate)
    topb = topb.join(dfbooks.set_index("book_id")[["title"]])
    topb = topb.sort_values("is_read", ascending=False).head(top_books)

    dff = dff[dff["user_id"].isin(topu.index)]
    dff = dff[dff["book_id"].isin(topb.index)]

    mat = dff.pivot(index="book_id", columns="user_id", values="rating").fillna(0)
    time_taken = time.time() - start
    logger.info(f"Finished generating user-rating matrix in {time_taken:.2f} seconds")
    return mat


if __name__ == "__main__":
    start = time.time()

    FNAME_MATRIX = Path("data").joinpath("matrix.snap.parquet")
    PATH_INTERACTIONS = Path("data").joinpath("goodreads_interactions.snap.parquet")
    PATH_BOOKS = Path("data").joinpath("books_extra_features.snap.parquet")
    PATH_TITLES = Path("data").joinpath("titles.snap.parquet")

    titles = pd.read_parquet(PATH_TITLES)
    DFI = pd.read_parquet(PATH_INTERACTIONS)
    DFB = pd.read_parquet(PATH_BOOKS)

    mat = create_matrix(dfi=DFI, dfbooks=DFB, top_books=TOP_BOOKS, top_users=TOP_USERS)

    mat.columns = [str(f) for f in mat.columns]
    mat.to_parquet(FNAME_MATRIX)
