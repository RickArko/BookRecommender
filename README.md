# BookRecommender
Repository for Book Recommendations using UCSD Goodreads Data.

## Setup:

### Data
Download GoodReads data from [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home).

- [books](https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK)
- [interactions-dataset](https://drive.google.com/u/0/uc?id=1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon&export=download) and save in `data/`
### Installation
```
    python -m venv venv
    venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python -m ipykernel install --user --name=book-rec
```

### Models
1. Collaborative Filtering
2. Content Based


### Resources
  - [collaborative-filtering-deep-dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive)