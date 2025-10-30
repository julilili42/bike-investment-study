from datasets import load_dataset
from pydantic import BaseModel, TypeAdapter
import pandas as pd
from itertools import islice
from datetime import date
from typing import Iterator
import numpy as np 
from typing import Sequence
from scipy import sparse
import matplotlib.pyplot as plt


class BaseArticle(BaseModel):
    date: date
    headline: str
    article: str

class Article(BaseArticle):
  short_headline: str
  short_text: str
  link: str

class Mail(BaseArticle):
  pass

def import_corpus(batch_size: int, url: str, adapter: TypeAdapter | None = None, num_batches: int | None = None, split: str ="train", streaming: bool = True) -> Iterator[Article]:
  ds = load_dataset(url, split=split, streaming=streaming)

  if not adapter: 
    adapter = TypeAdapter(list[Article])

  batch_counter = 0

  if not streaming: 
    print(num_batches * batch_size if num_batches else len(ds), "articles to be loaded for dataset", ds.info.dataset_name)

  # streaming -> iterator, in-memory -> dataset
  ds = ds if streaming else iter(ds)

  # continuously iterating data by given batch_size saves ram, enables limited output by num_batches parameter
  while True:
    if num_batches and batch_counter >= num_batches:
      break

    batch = list(islice(ds, batch_size))

    if not batch:
      break
    
    batch_counter += 1
    # validating the structure of the json batch enables easy access of the fields
    yield from adapter.validate_python(batch) 

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


def linear_prediction(col, target_year=2024, eps=1e-6):
    y = col.dropna()
    if len(y) < 2: 
        return float(y.iloc[-1]) if len(y) else eps
    x = y.index.values.astype(float)
    y_ = np.clip(y.values, eps, 1 - eps)
    z = np.log(y_ / (1 - y_))
    m, b = np.polyfit(x, z, 1)
    zhat = m * target_year + b
    phat = 1 / (1 + np.exp(-zhat))
    return float(phat)



# plot occurency frequency to years
def plot_word_timeseries(
    word: str,
    occurence_freq: pd.DataFrame,
    occurrences_abs: pd.DataFrame | None = None,
    train_years=(2020, 2022),
    target_year=2024,
    show=True,
):
    if word not in occurence_freq.columns:
        raise ValueError(f"Wort '{word}' nicht im Vokabular.")

    series = occurence_freq[word].dropna()
    years_all = series.index.values.astype(int)

    y0, y1 = train_years
    train = series.loc[(series.index >= y0) & (series.index <= y1)]

    q = linear_prediction(train, target_year=target_year)

    x_train = train.index.values.astype(float)
    m, b = np.polyfit(x_train, train.values, 1)
    
    years_trend = np.arange(y0, target_year + 1)
    trend_vals = m * years_trend + b

    _, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(6, 3), dpi=160, gridspec_kw={'width_ratios': [2.3, 1]})
    ax = ax_plot

    ax.plot(years_all, series.values, marker='o')

    ax.plot(years_trend, trend_vals, linestyle='--')

    p = series.get(target_year, np.nan)

    y_min = min(series.min(), np.nanmin(trend_vals))
    y_max = max(series.max(), np.nanmax(trend_vals))
    y_margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(max(0, y_min - y_margin), y_max + y_margin)

    ax.set_title(word)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Frequenz")
    ax.legend()

    ax_text.axis("off")

    text_lines = []

    if occurrences_abs is not None and word in occurrences_abs.columns:
      abs_counts = occurrences_abs[word].dropna()   
      freq_series = occurence_freq[word].dropna()   

      text_lines.append("Occurrences per Year:\n")
      total_occ = 0

      years = abs_counts.index.intersection(freq_series.index)
      for year in years:
          abs_value = int(abs_counts.loc[year])
          freq_value = float(freq_series.loc[year])
          total_occ += abs_value
          text_lines.append(f"{year}: {abs_value:>6}   {freq_value:>8.3%}")

      text_lines.append("")
      text_lines.append(f"Total: {int(total_occ):>6}")
      text_lines.append("")

    ax_text.text(
        0, 1, "\n".join(text_lines),
        va="top", ha="left", fontsize=9, family="monospace"
    )

    if show:
        plt.tight_layout()
        plt.show()

    return {"p": float(p) if np.isfinite(p) else None,
            "q": float(q),
            "delta": float((p - q)) if np.isfinite(p) else None}
