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

# SAMPLE PARAMS
MIN_READS = 50
TOP_BOOKS = 1_000
TOP_USES = 50_000

# contains some one-time data processing
# # reals all 228 million user-book interactions
# dfi = pd.read_feather('data/full_interactions.feather')
# # limit only interactions with known books
# dfi = dfi.merge(dfb[['book_id']])
# # 228 million (with 2.3 million books) to 88 million (with 438k books)
# dfi.to_feather('data/interactions.feather')


# load data
# dft = pd.read_feather('data/titles.feather')
# dfb = pd.read_feather('data/simple_book_features.feather')
# dfc = pd.read_feather('data/books_extra_features.feather')


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


## Load/Sample data
start = time.time()
dfi = pd.read_feather('data/interactions.feather')
dff = active_filter(dfi, min_reads=MIN_READS)
fname = f'data/filter/df_filter_min_reads-{min_reads}.feather'
time_taken = time.time() - start


### Select a representative sample to build off
## grab 10k from popular online set
## supplement with top 1000k by straight interactions

df_books = pd.read_csv('data/df_books.csv')
sample_book_ids = df_books['goodreads_book_id']

top_x_book_ids = get_top_x_books(dff, 1000)
# sample_books_ids
# randomly sample 8_000

user_ids = get_top_x_users(dff, 75_000)
book_ids = set(set(sample_book_ids.values).union(top_x_book_ids))

# template with normalized book_id and title
dftmp = dft.loc[dft['book_id'].isin(book_ids)].reset_index(drop=True)
dftmp['goodreads_book_id'] = dftmp['book_id']
dftmp['goodreads_book_id'] = dftmp.sort_values('book_id')
dftmp['book_id'] = range(dftmp.shape[0])
dftmp.set_index('book_id')
dftmp.to_pickle('data/filter/filtered_titles.pkl')

dff.rename(columns={'book_id':'goodreads_book_id'}, inplace=True)
df_ = dftmp.merge(dff, right_on=['good_readsbook_id'], left_on=['goodreads_book_id'])
# df_.to_feather('data/filter/all_interactions.feather')

is_read = (df_.is_read == 1)
is_top_user = (df_.user_id.isin(user_ids))

dff = df_.loc[is_read & is_top_user]

book_user_mat = dff.pivot(
    index='book_id', columns='user_id', values='rating').fillna(0)

book_to_idx = {
    book: i for i, book in 
    enumerate(list(dftmp.loc[book_user_mat.index].title))
}


# dffilter = dff.loc[dff.book_id.isin(top_x_book_ids)]

## will need to down sample do to time constraints
# recommend based on top 10k books by popularity + 750 
# by interactions

df_filter = dff.loc[dff['book_id'].isin(book_ids)]
df_filter = df_filter.loc[df_filter['user_id'].isin(top_user_ids)]
df_filter.shape  # ~50 million interactions

# ## Item Based Recommendation with KNN

# # Format data for user-item recommenations
# # Transform df_ratings (dfi) into an (m x n) array
# # m (# books)
# # n (# users)

start = time.time()
book_user_matrix = df_filter.pivot(
            index='book_id', columns='user_id', values='rating').fillna(0)

# book_user_matrix.to_pickle('data/filter/book_user_matrix_.pkl')
seconds = time.time() - start


dft.set_index('book_id').loc[matrix.index]
# dfi.merge(df_books.rename('good_reads_book_id': 'book_id'), on=[])
dfq = df_books.drop('book_id', 1).merge(dfi, left_on=['goodreads_book_id'], right_on=['book_id'])
dft.set_index('book_id').loc[sample_book_ids]


start = time.time()
book_user_matrix = dfq.pivot(
            index='book_id', columns='user_id', values='rating').fillna(0)

# book_user_mat.to_pickle('data/filter/book_user_matrix_.pkl')
seconds = time.time() - start




# think
dfr = dfi.loc[dfi.is_read]

#### most basic
df_books = pd.read_csv('data\df_books.csv')
df_ratings = pd.read_csv('data\df_ratings.csv')

book_user_mat = df_ratings.pivot(index='book_id', columns='user_id', values='rating').fillna(0)

book_to_idx = {
    book: i for i, book in 
    enumerate(list(df_books.set_index('book_id').loc[book_user_mat.index].title))
}




sparse_matrix = csr_matrix(book_user_mat.values)

model_knn = NearestNeighbors(metric='cosine', 
                             algorithm='brute', 
                             n_neighbors=20, 
                             n_jobs=-1)

model_knn.fit(sparse_matrix)



book_name = 'Harry Botter and the Chamber of Secrets'
idx = fuzzy_matching(book_to_idx, book_name)


### sampling technique
# revisit
# df_books = pd.read_csv('data/df_books.csv')
# sample_book_ids = df_books['goodreads_book_id']
dfbooks = dft.loc[dft['book_id'].isin(sample_book_ids)]

top1000books = get_top_x_books(dfi, 1000)
# # recommend based on top 10k books by popularity
# top_x = 750
# top_ids = get_top_x_books(dff)


df = dff