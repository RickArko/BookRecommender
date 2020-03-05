"""Model to make simple KNN prediction
"""
import os
import time
import gc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz

# processed data impports
matrix = pd.read_pickle('data/filter/book_user_matrix_.pkl')
DFTMP = pd.read_pickle('data/filter/filtered_titles.pkl')
KNN20 = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

hashmap = {
            book: i for i, book in
            enumerate(list(DFTMP.set_index('book_id').loc[matrix.index].title)) # noqa
        }

DFTMP.loc[matrix.index].title

DEFAULT_TOP_BOOKS = [536,  943, 1000, 1387,  968,  941, 1386,  938, 1473, 1402,  461,
                     586,  996, 1116,  759,   66, 1203,  524, 1007, 7008]

def fuzzy_matching(mapper, fav_book, verbose=True):
    """Return book index of closest match via fuzzy ratio. 
    If no match found, return None

    inputs:
    ------
        mapper:   dict, {title: index of the book in data}
        fav_book: str
        verbose:  bool

    return:
    ------
        index of the closest match
    """
    match_tuple = []

    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), book.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))

    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        if verbose:
            print('No match.')
        return
    if verbose:
        print(f'possible matches: {[x[0] for x in match_tuple]}')
    return match_tuple[0][1]



class KnnRecommender:
    """Item-based Collaborative Filtering Recommender
    """
    def __init__(self, matrix, hashmap, model=KNN20):
        """Initialize Class
        """
        self.data = matrix
        self.hashmap = hashmap
        self.model = model

    def _train_model(self):
        start = time.time()
        self.model.fit(csr_matrix(self.data))
        msg = f'Fit model in {time.time() - start} seconds'
        print(msg)
        return self

    def make_recommendation_from_book(self, book_name, n_recommendations):
        """Give top n_recommendations books based on input
        book_name
        """
        distances, indices = self.model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations+1)

        # sort recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]

        reverse_hashmap = {v: k for k, v in self.hashmap.items()}
        print('Recommendations for {}:'.format(book_name))
        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i+1, reverse_hashmap[idx], dist))



if __name__ == '__main__':
    start = time.time()
    rec = KnnRecommender(matrix, hasmap)
    rec.make_recommendations('Harry Pottter and the chamber of secrets', n_recommendations=10)
    seconds_taken = time.time() - start