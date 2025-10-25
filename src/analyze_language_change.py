from datasets import load_dataset
from pydantic import BaseModel, TypeAdapter
import pandas as pd
from itertools import islice
from datetime import date
from typing import Iterator
import numpy as np 
from typing import Sequence
from scipy import sparse


class Article(BaseModel):
  date: date
  headline: str
  short_headline: str
  short_text: str
  article: str
  link: str

def import_corpus(batch_size: int, url: str, num_batches: int | None = None, split: str ="train", streaming: bool =True) -> Iterator[Article]:
  ds = load_dataset(url, split=split, streaming=streaming)

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


def linear_prediction(col: pd.Series, target_year: int=2024) -> float:
    y = col.dropna()
    x = y.index.values.astype(float)
    
    # linear fit 
    m, b = np.polyfit(x, y.values, deg=1)

    pred = m * target_year + b
    
    # clamp to [0,1]
    return float(np.clip(pred, 0.0, 1.0))
