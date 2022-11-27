import os

import gdown
from loguru import logger


DATA_DIR = "data"
DOWNLOAD_MAP = {
    # "1TLmSvzHvTLLLMjMoQdkx6pBWon-4bli7": "goodreads_book_works.json.gz",
    # "19cdwyXwfXx_HDIgxXaHzH0mrx8nMyLvC": "goodreads_book_authors.json.gz",
    # "1op8D4e5BaxU2JcPUgxM3ZqrodajryFBb": "goodreads_book_series.json.gz",
    "1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK": "goodreads_books.json.gz",
    # "1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md": "goodreads_book_genres_initial.json.gz",
    # "1R3wJPgyzEX9w6EI8_LmqLbpY4cIC9gw4": "goodreads_books_children.json.gz",
    # "1ICk5x0HXvXDp5Zt54CKPh5qz1HyUIn9m": "goodreads_books_comics_graphic.json.gz",
    # "1x8IudloezYEg6qDTPxuBkqGuQ3xIBKrt": "goodreads_books_fantasy_paranormal.json.gz",
    # "1roQnVtWxVE1tbiXyabrotdZyUY7FA82W": "goodreads_books_history_biography.json.gz",
    # "1ACGrQS0sX4-26D358G2i5pja1Y6CsGtz": "goodreads_books_mystery_thriller_crime.json.gz",
    # "1H6xUV48D5sa2uSF_BusW-IBJ7PCQZTS1": "goodreads_books_poetry.json.gz",
    # "1juZreOlU4FhGnBfP781jAvYdv-UPSf6Q": "goodreads_books_romance.json.gz",
    # "1gH7dG4yQzZykTpbHYsrw2nFknjUm0Mol": "goodreads_books_young_adult.json.gz",
    # "1Cf90P5TH84ufrs8qyLrM-iWOXOGjBi9r": "goodreads_interactions_children.json.gz",
    # "1CCj-cQw_mJLMdvF_YYfQ7ibKA-dC_GA2": "goodreads_interactions_comics_graphic.json.gz",
    # "1EFHocJIh5nknbUMcz4LnrMEJkwW3Vk6h": "goodreads_interactions_fantasy_paranormal.json.gz",
    # "10j181giCD94pcYynd6fy2U0RyAlL66YH": "goodreads_interactions_history_biography.json.gz",
    # "1xuujDT-vOMMkk2Kog0CTT9ADmnD8pa9D": "goodreads_interactions_mystery_thriller_crime.json.gz",
    # "17G5_MeSWuhYnD4fGJMvKRSOlBqCCimxJ": "goodreads_interactions_poetry.json.gz",
    # "1OmPKA0TmL0nnECDRNF1YpWJw6PBDFl_j": "goodreads_interactions_romance.json.gz",
    # "1NNX7SWcKahezLFNyiW88QFPAqOAYP5qg": "goodreads_interactions_young_adult.json.gz",
    # "1908GDMdrhDN7sTaI_FelSHxbwcNM1EzR": "goodreads_reviews_children.json.gz",
    # "1V4MLeoEiPQdocCbUHjR_7L9ZmxTufPFe": "goodreads_reviews_comics_graphic.json.gz",
    # "1THnnmE4XSCvMkGOsqapQr2cJI5amKA6X": "goodreads_reviews_fantasy_paranormal.json.gz",
    # "1lDkTzM6zpYU-HGkVAQgsw0dBzik-Zde9": "goodreads_reviews_history_biography.json.gz",
    # "1ONpyuv0vrtd6iUEp0-zzkKqwpm3njEqi": "goodreads_reviews_mystery_thriller_crime.json.gz",
    # "1FVD3LxJXRc5GrKm97LehLgVGbRfF9TyO": "goodreads_reviews_poetry.json.gz",
    # "1NpFsDQKBj_lrTzSASfyKbmkSykzN88wE": "goodreads_reviews_romance.json.gz",
    # "1M5iqCZ8a7rZRtsmY5KQ5rYnP9S0bQJVo": "goodreads_reviews_young_adult.json.gz",
    "1CHTAaNwyzvbi1TR08MJrJ03BxA266Yxr": "book_id_map.csv",
    "15ax-h0Oi_Oyee8gY_aAQN6odoijmiz6Q": "user_id_map.csv",
    "1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon": "goodreads_interactions.csv",
    "1pQnXa7DWLdeUpvUFsKusYzwbA5CAAZx7": "goodreads_reviews_dedup.json.gz",
    "196W2kDoZXRPjzbTjM6uvTidn6aTpsFnS": "goodreads_reviews_spoiler.json.gz",
    "1NYV4F1WGJg6QbV0rOSXi6Y1gFLwic94a": "goodreads_reviews_spoiler_raw.json.gz",
}


def download_file(id, output=DATA_DIR, quiet=False):
    """Download a file from Google Drive by its id.

    Args:
        id (_type_): File id.
        output (_type_, optional): _description_. Defaults to DATA_DIR.
        quiet (bool, optional): _description_. Defaults to False.
    """
    url = f"https://drive.google.com/uc?id={id}"
    gdown.download(url, output=output, quiet=quiet)


def download_goodreads_data(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    for key, name in DOWNLOAD_MAP.items():
        logger.info(f"Downloading {name}")
        download_file(key, output=data_dir)


if __name__ == "__main__":
    download_goodreads_data(DATA_DIR)