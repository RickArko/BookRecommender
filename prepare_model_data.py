"""Module to prepare data for modelling
Sources:
- interactions
- books string features
- books numeric features

- full_interactions.feather - 228 million reader-book interactions
- interactions.feather - 88 million reader-book interactions (with known books)

- `simple_book_features.feather` - numberic book feats
"""
import datetime
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# contains some one-time data processing

# reals all 228 million user-book interactions
dfi = pd.read_feather('data/full_interactions.feather')
# limit only interactions with known books
dfi = dfi.merge(dfb[['book_id']])
# 228 million (with 2.3 million books) to 88 million (with 438k books)
dfi.to_feather('data/interactions.feather')


# dft = dft = pd.read_feather('data/titles.feather')
dft = pd.read_feather('data/titles.feather')
dfb = pd.read_feather('data/simple_book_features.feather')
dfc = pd.read_feather('data/books_extra_features.feather')


def active_filter(data, min_reads=50):
    """Filter interactions data by activity theshold
    (min_reads)
    """
    df_user_counts = pd.DataFrame(data.groupby('user_id').size(), columns=['count'])
    df_book_counts = pd.DataFrame(data.groupby('book_id').size(), columns=['count'])

    active_users = df_user_counts[df_user_counts['count'] >= min_reads].index.values
    active_user_filter = data['user_id'].isin(active_users).values

    popular_books = df_book_counts[df_book_counts['count'] >- min_reads].index.values
    popular_books_filter = data['book_id'].isin(popular_books).values

    return data[active_user_filter & popular_books_filter]


def get_top_x_users(data, top_x=10_000):
    """Get top (x) users by interactions
    returns: array of book_ids
    """
    df_user_counts = pd.DataFrame(data.groupby('user_id').size(), columns=['count'])
    return df_user_counts.sort_values('count', ascending=False).head(top_x).index.values


def get_top_x_books(data, top_x=10_000):
    """Get top (x) books by interactions
    returns: array of book_ids
    """
    df_book_counts = pd.DataFrame(data.groupby('book_id').size(), columns=['count'])
    return df_book_counts.sort_values('count', ascending=False).head(top_x).index.values



# # recommend based on top 10k books by popularity
# top_x = 750
# top_ids = get_top_x_books(dff)
# df_filter = dff.loc[dff['book_id'].isin(top_ids)]
# df_filter.shape  # ~50 million interactions

# start = time.time()
# book_user_matrix = df_filter.pivot(
#             index='book_id', columns='user_id', values='rating').fillna(0)

# # book_user_matrix.to_pickle('data/fileter/book_user_matrix_.pkl')
# seconds = time.time() - start


# dfi = pd.read_feather('data/interactions.feather')
# dff = active_filter(dfi)
# fname = f'data/df_filter_min_reads-{min_reads}.feather'
# df_filter.to_feather(fname)

### sampling technique
# revisit
# df_books = pd.read_csv('data/df_books.csv')
# book_ids = set(ids[0:750]).union(set(df_books.goodreads_book_id))

## will need to down sample do to time constraints
# recommend based on top 10k books by popularity + 750 
# by interactions

df_filter = dff.loc[dff['book_id'].isin(book_ids)]
df_filter = df_filter.loc[df_filter['user_id'].isin(top_user_ids)]
df_filter.shape  # ~50 million interactions

start = time.time()

# ## Item Based Recommendation with KNN

# # Format data for user-item recommenations
# # Transform df_ratings (dfi) into an (m x n) array
# # m (# books)
# # n (# users)

book_user_matrix = df_filter.pivot(
            index='book_id', columns='user_id', values='rating').fillna(0)

# book_user_matrix.to_pickle('data/filter/book_user_matrix_.pkl')
seconds = time.time() - start

