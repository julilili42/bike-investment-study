from datasets import load_dataset
from pydantic import BaseModel, TypeAdapter
import pandas as pd
from itertools import islice, chain
from sklearn.feature_extraction.text import CountVectorizer
from datetime import date
from typing import Iterator
import multiprocessing as mp
import numpy as np 
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


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



def linear_prediction(col: pd.Series, year_of_prediction: int=2024) -> float:
    y = col.dropna()
    x = y.index.values.astype(float)
    
    # linear fit 
    m, b = np.polyfit(x, y.values, deg=1)

    pred = m * year_of_prediction + b
    
    return float(np.clip(pred, 0.0, 1.0))



def main():
  corpus_18_23 = import_corpus(batch_size=100, streaming=False, url="bjoernp/tagesschau-2018-2023")
  corpus_24 = import_corpus(batch_size=100, streaming=False, url="bjoernp/tagesschau-010124-020524")

  corpus = chain(corpus_18_23, corpus_24)

  rows = [(a.date.year, a.article) for a in corpus]
  year, articles = zip(*rows)

  # generate word matrix 
  vec = CountVectorizer(max_features=100000)
  word_matrix = vec.fit_transform(articles)
  vocab = vec.get_feature_names_out()

  word_matrix_bin = word_matrix.sign()

  unique_years = np.unique(year)
  
  # distribution of articles in unique years
  year_distribution = defaultdict(int)

  # dimension unique_years x vocab
  rows = []

  for y in unique_years:
      sel = (year == y)
      # number of articles in year y
      n = int(sel.sum())
      year_distribution[int(y)] = n
      
      # count how many articles (of this year) contain each word at least once -> dimension 1 x n 
      present_counts = np.asarray(word_matrix_bin[sel].sum(axis=0)).ravel()
      rows.append(present_counts / max(n, 1))
  
  # occurence_freq per year
  occurence_freq = pd.DataFrame(rows, index=unique_years, columns=vocab)

  
  # goal: try to find words which disproportionaly increased in use 
  # dataframe containing frequencies before introduction of llm's
  train = occurence_freq.loc[2018:2022]
  
  # linear interpolation, frequency value of 2024
  q = train.apply(linear_prediction, axis=0)

  # empirical frequency
  p = occurence_freq.loc[2024]

  # Metric 1: frequency gap delta = p - q
  delta = p - q

  # Metric 2: frequency ratio r = p/q
  r = p / q.replace(0, 1e-8)
  sorted_freq_gap = delta.sort_values(ascending=False).head(100)
  sorted_freq_ratio = r.sort_values(ascending=False).head(100)

  print(sorted_freq_gap)
  print(sorted_freq_ratio)

  
if __name__ == "__main__":
  mp.set_start_method("fork", force=True)
  main()