from transformers import AutoModel, AutoTokenizer
import torch
import argparse
from functools import partial
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch
from rank_bm25 import BM25Okapi
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
model = AutoModel.from_pretrained("facebook/contriever").to('cuda')

title_to_text=None
entity_to_titles=None

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def encode_batch(data, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)


def calculate_recall(list_searchspace, search_space, list_golden, topks, query, retriever):
    if retriever == "contriever":
        with torch.no_grad():
            query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to('cuda')
            query_embedding = mean_pooling(model(**query_inputs).last_hidden_state,query_inputs['attention_mask']).cpu()  # Move to CPU
            query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
            search_space = search_space / search_space.norm(dim=1, keepdim=True)
            similarities = torch.matmul(search_space, query_embedding.T).squeeze()
            ranked_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        
    elif retriever == "bm25":
        tokenized_query = query.split(" ")
        bm25_scores = search_space.get_scores(tokenized_query)
        ranked_indices = np.argsort(bm25_scores)[::-1]
    
    else:
        raise ValueError("Invalid retriever type. Choose 'bm25' or 'contriever'")
    
    ls_re = []
    for topk in topks:
        topk_indices = ranked_indices[:topk]
        retrieved_elements = {list_searchspace[i] for i in topk_indices}
        ls_re.append(retrieved_elements)

    return ls_re

def map_recall(row, topks,_type,list_search_space,search_space,retriever):
    row['golden_title'] = [t['wikipedia_title'] for t in row['contexts'] if t['is_supporting']]
    row['golden_text'] = [t['wikipedia_title']+" "+t['paragraph_text'] for t in row['contexts'] if t['is_supporting']]
    if(_type=="baseline"):
        ls_re =calculate_recall(list_search_space, search_space,row['golden_text'],topks,row['question_text'],retriever)
        recalls = []
        for tmp in ls_re:
            relevant_retrieved = tmp.intersection(set(row['golden_text']))
            recall = len(relevant_retrieved) / len(row['golden_text']) if row['golden_text'] else 0
            recalls.append(recall)
        
        row['recall']=recalls

    else:
        ls_re=calculate_recall(list_search_space, search_space,row['golden_title'],topks,row['question_text'],retriever)
    
        recalls = []
        counts = []
        for tmp in ls_re:
            titles=[]
            for entity in tmp:
                titles+=(entity_to_titles[entity])
            titles=set(titles)
            relevant_retrieved = titles.intersection(set(row['golden_title']))
            recall = len(relevant_retrieved) / len(row['golden_title']) if row['golden_title'] else 0
            recalls.append(recall)
            counts.append(len(titles))
        row['recall']=recalls
        row['count']=counts

    return row


def format_dataset(row,dataset_type):
    if(dataset_type == "musique"):
        return row
    elif(dataset_type == "hotpotqa"):
        row["question"]= row["question_text"]
        row["answer"]=row["answer_text"]
        row["paragraphs"]=[{"idx":tmp["wikipedia_id"],"title":tmp["wikipedia_title"],"is_supporting":tmp["is_supporting"],"paragraph_text":tmp["paragraph_text"]}for tmp in row["contexts"]]
        return row
    elif(dataset_type == "2wikimultihopqa"):
        row["question"]= row["question_text"]
        row["answer"]=row["answer_text"]
        row["paragraphs"]=[{"idx":tmp["wikipedia_id"],"title":tmp["wikipedia_title"],"is_supporting":tmp["is_supporting"],"paragraph_text":tmp["paragraph_text"]}for tmp in row["contexts"]]
        return row
    else:
        assert("dataset type not recognized")
    
def get_dataset(dataset_path,dataset_type):
    if(dataset_type == "musique"):
        dataset = load_dataset('json', data_files=dataset_path)["train"] 
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    elif(dataset_type == "hotpotqa"):
        dataset = load_dataset('json', data_files=dataset_path)["train"] 
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    elif(dataset_type == "2wikimultihopqa"):
        with open(dataset_path,"r") as f:
            a = list(f)
        x=[json.loads(t) for t in a]
        dataset = Dataset.from_list(x)
        return dataset.map(partial(format_dataset,dataset_type=dataset_type))
    else:
        assert("dataset type not recognized")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Retrieval Test")
    parser.add_argument("--original_dataset_path", type=str, default="data/2wikimultihopqa_dev_20k.jsonl", help="model name for evaluation")
    parser.add_argument("--db_path", type=str, default="tkdrnjs0621/db_2wikimultihopqa_dev", help="model name for evaluation")
    parser.add_argument("--db2_path", type=str, default="tkdrnjs0621/db_2wikimultihopqa_dev_entity", help="model name for evaluation")
    parser.add_argument("--dataset_type", type=str, default="2wikimultihopqa", help="model name for evaluation")
    parser.add_argument("--retriever", type=str, default="bm25", help="model name for evaluation")
    parser.add_argument("--type", type=str, default="baseline", help="model name for evaluation")
    parser.add_argument("--topks", type=str, default="1,2,3,4,5,6,7,8,9,10", help="batch size for inference")
    parser.add_argument("--save_path", type=str, default="inference/2wikimultihopqa/ner", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    
    args = parser.parse_args()

    args.topks = [int(x) for x in args.topks.split(',')]
    
    original_dataset = get_dataset(args.original_dataset_path, args.dataset_type)
    a1 = original_dataset.filter(lambda x:x['other_info']['type']=='comparison').select(range(250))
    a2 = original_dataset.filter(lambda x:x['other_info']['type']=='compositional').select(range(250))
    a3 = original_dataset.filter(lambda x:x['other_info']['type']=='inference').select(range(250))
    a4 = original_dataset.filter(lambda x:x['other_info']['type']=='bridge_comparison').select(range(250))
    original_dataset = concatenate_datasets([a1,a2,a3,a4])
    print(len(original_dataset))

    db = load_dataset(args.db_path, split="train")
    title_to_text = {t["title"]:t["text"] for t in db}
    db2 = load_dataset(args.db2_path,split="train")
    entity_to_titles = {t["entity"]:t["titles"] for t in db2}

    if args.type=='baseline':
        list_search_space = [t['text'] for t in db]
        if args.retriever=="bm25":
            tokenized_searchspace = [t['text'].split(" ") for t in db]
            bm25 = BM25Okapi(tokenized_searchspace)
            new_dataset = original_dataset.map(partial(map_recall,topks=args.topks,_type=args.type,list_search_space=list_search_space,search_space=bm25,retriever=args.retriever))
        else:
            data_embeddings = encode_batch([t['text'] for t in db])
            new_dataset = original_dataset.map(partial(map_recall,topks=args.topks,_type=args.type,list_search_space=list_search_space,search_space=data_embeddings,retriever=args.retriever))
            
    else:
        list_search_space = [t['entity'] for t in db2]
        if args.retriever=="bm25":
            tokenized_searchspace = [t['entity'].split(" ") for t in db2]
            bm25 = BM25Okapi(tokenized_searchspace)
            new_dataset = original_dataset.map(partial(map_recall,topks=args.topks,_type=args.type,list_search_space=list_search_space,search_space=bm25,retriever=args.retriever))
        else:
            data_embeddings = encode_batch([t['entity'] for t in db2])
            new_dataset = original_dataset.map(partial(map_recall,topks=args.topks,_type=args.type,list_search_space=list_search_space,search_space=data_embeddings,retriever=args.retriever))

    
    args.save_path = f"inferece/{args.dataset_type}_{args.type}_{args.retriever}"

    new_dataset.save_to_disk(args.save_path)
    
    array = np.array(new_dataset['recall'])
    avg = array.mean(axis=0)

    print("Recall",avg)
    if args.type!="baseline":
        array = np.array(new_dataset['count'])
        avg = array.mean(axis=0)
        print("Count",avg)
