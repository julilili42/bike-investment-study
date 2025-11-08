from datasets import load_dataset
from itertools import islice
from typing import Iterator
from pydantic import TypeAdapter
from .models import Paper

def import_corpus(batch_size: int, url: str, adapter: TypeAdapter | None = None, num_batches: int | None = None, split: str ="train", streaming: bool = True) -> Iterator[Paper]:
  ds = load_dataset(url, split=split, streaming=streaming)

  if not adapter: 
    adapter = TypeAdapter(list[Paper])

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