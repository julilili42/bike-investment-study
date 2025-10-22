from datasets import load_dataset
from pydantic import BaseModel, TypeAdapter
import pandas as pd
from itertools import islice
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

  # streaming -> iterator, in-memory -> dataset
  ds = ds if streaming else iter(ds)

  print(num_batches * batch_size if num_batches else 21900, "articles to be loaded")

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




def main():
  corpus_18_23 = import_corpus(batch_size=100, streaming=False, url="bjoernp/tagesschau-2018-2023")
  
  rows = [(a.date.year, a.article) for a in corpus_18_23]
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

  # frequency gap 
  y0, yN = occurence_freq.index[0], occurence_freq.index[-1]
  freq_gap = occurence_freq.loc[yN] - occurence_freq.loc[y0]

  # top 20 words with largest increase of frequency from y0 to yN
  sorted_freq_gap = freq_gap.sort_values(ascending=True).head(20)

  print(sorted_freq_gap)

  df_year = pd.DataFrame(
    list(year_distribution.items()), columns=["year", "count"]
  ).sort_values("year")

  sns.barplot(data=df_year, x="year", y="count")
  plt.xticks(rotation=45, ha="right")
  plt.title("Artikel pro Jahr")
  plt.tight_layout()
  plt.show()


  
if __name__ == "__main__":
  mp.set_start_method("fork", force=True)
  main()