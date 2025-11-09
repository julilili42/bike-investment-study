import numpy as np
import pandas as pd
from scipy import sparse
from typing import Sequence

"""
produces years x vocab occurrence frequencies matrix
"""
def occurrence_freq_by_year(
    year: np.ndarray,                  # shape: year
    word_matrix_bin: sparse.spmatrix,  # shape: year x vocab, binary (>=1 -> 1)
    vocab: Sequence[str]
) -> pd.DataFrame:
    unique_years = np.sort(np.unique(year))
    # dimension unique_years x vocab
    rows = []
    for y in unique_years:
        sel = (year == y)
        # number of articles in year y
        n = int(sel.sum())
        # count how many articles (of this year) contain each word at least once -> dimension 1 x n 
        present_counts = np.asarray(word_matrix_bin[sel].sum(axis=0)).ravel()  
        rows.append(present_counts / max(n, 1))
    return pd.DataFrame(rows, index=unique_years, columns=vocab)


def occurrences_by_year(
    year: np.ndarray,                 # shape: n_docs
    word_matrix: sparse.spmatrix,     # shape: n_docs x vocab
    vocab: Sequence[str],             
) -> pd.DataFrame:
    word_matrix = word_matrix.tocsr()

    years = np.asarray(year)
    unique_years = np.sort(np.unique(years))

    rows = []
    for y in unique_years:
        sel = (years == y)
        counts = np.asarray(word_matrix[sel].sum(axis=0)).ravel()
        rows.append(counts)

    return pd.DataFrame(rows, index=unique_years, columns=vocab)


def occurences_by_word(word, year, articles):
    mask = [word in t.lower() for t in articles]
    counts = pd.Series(year)[mask].value_counts().sort_index()
    return counts