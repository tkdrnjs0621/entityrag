from datasets import load_from_disk, Dataset
import json
from datasets import load_dataset
import argparse
from functools import partial
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi
import numpy as np

#1 Loading NER Data
original_dataset=[]
ner_result = load_from_disk("/home/tkdrnjs0621/work/entityrag/inference/2wikimultihopqa/ner")

with open("./data/2wikimultihopqa_dev_20k.jsonl", "r") as f:
    for x in f.readlines():
        original_dataset.append(json.loads(x))
# original_dataset = load_from_disk("/home/tkdrnjs0621/work/entityrag/db/2wikimultihopqa_1000")

#2 Building title based DB
large_data = []
for data in original_dataset:
    for t in data["contexts"]:
        large_data.append({"title":t["wikipedia_title"],"text":t["wikipedia_title"]+" "+t["paragraph_text"]})
db = Dataset.from_list(large_data)

ner_dict = {}
for p in ner_result:
    ner_dict[p['title']] = p['output'].strip().split('\n')
def map_t(row):
    row['entities']=ner_dict[row['title']]
    return row
db = db.map(map_t)
db.push_to_hub("tkdrnjs0621/db_2wikimultihopqa_dev",split="train")

#3 Building Entity based DB
new_db = {}
for row in db:
    for e in row['entities']:
        if not e in new_db:
            new_db[e]=[]
        new_db[e].append(row['title'])
new_db = [{"entity":k,"titles":list(set(v))} for k,v in new_db.items()]
new_db = Dataset.from_list(new_db)
new_db.push_to_hub("tkdrnjs0621/db_2wikimultihopqa_dev_entity")