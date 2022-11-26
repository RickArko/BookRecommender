import codecs
import time
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix
from tqdm import tqdm

FNAME_TITLES = Path("data").joinpath("titles.snap.parquet")
FNAME_MATRIX = Path("data").joinpath("matrix.snap.parquet")
FNAME_RESULTS = Path("data").joinpath("cosine-similarity.txt")

if __name__ == "__main__":

    mat = pd.read_parquet(FNAME_MATRIX)
    mat.columns = [int(c) for c in mat.columns]
    csr_matrix = csr_matrix(mat.astype(pd.SparseDtype("float64", 0)).sparse.to_coo())

    titles = pd.read_parquet(FNAME_TITLES)
    TITLE_MAP = titles.set_index("book_id")["title"].to_dict()

    start = time.time()
    model = CosineRecommender(K=100, num_threads=-1)
    model.fit(csr_matrix)

    with tqdm(total=len(mat)) as progress:
        with codecs.open(FNAME_RESULTS, "w", "utf8") as o:
            for _id in mat.index:
                title = TITLE_MAP.get(_id)
                for book, score in zip(*model.similar_items(_id, 11)):
                    o.write(f"{title}\t{TITLE_MAP.get(book)}\t{score}\n")
                progress.update(1)
