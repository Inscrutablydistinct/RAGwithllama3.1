import warnings
from datetime import datetime
from rank_bm25 import BM25Okapi
from model_param import CFG
from langchain_community.vectorstores import FAISS

warnings.filterwarnings("ignore")

def tokenize(text):
    return text.lower().split()

def compute_bm25_score(corpus, query):
    print(f"{corpus}\n")
    print(query)
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    return scores

def filter_attributes(metadata_entry, key, value, bm25_store):
    if key in ['title', 'author', 'abstract', 'keywords', 'results']:
        field_text = metadata_entry.get(key, "")
        print(field_text)
        bm25 = bm25_store.get(key)
        if bm25 is None:
            return 0.0
        tokenized_query = tokenize(value)
        scores = bm25.get_scores(tokenized_query)
        index = bm25.doc_freqs[0].index(field_text.lower()) if field_text.lower() in bm25.doc_freqs[0] else -1
        return scores[index] * 5 if index != -1 else 0.0
    elif key == 'publication_date':
        op = value[0] if value[1].isdigit() else value[0:2]
        value = value[len(op):]
        filter_date = datetime.strptime(value, "%Y-%m-%d")
        if metadata_entry['publication_date'] == "N/A":
            return 0.0
        entry_date = datetime.strptime(metadata_entry['publication_date'], "%Y-%m-%d")
        if op == '>':
            return 2.0 if entry_date > filter_date else -6.0
        elif op == '>=':
            return 2.0 if entry_date >= filter_date else -6.0
        elif op == '<':
            return 2.0 if entry_date < filter_date else -6.0
        elif op == '<=':
            return 2.0 if entry_date <= filter_date else -6.0
        else:
            return 2.0 if entry_date == filter_date else -6.0
    else:
        return 0.0

def filter_data(metadata, filter_dict):
    scored_metadata = []
    bm25_store = {}
    for key in filter_dict.keys():
        if key != 'publication_date':
            corpus = [entry.get(key, "") for entry in metadata]
            tokenized_corpus = [tokenize(doc) for doc in corpus]
            bm25_store[key] = BM25Okapi(tokenized_corpus)

    for entry in metadata:
        total_score = 0.0
        for key, value in filter_dict.items():
            total_score += filter_attributes(entry, key, value, bm25_store)
        print(total_score)
        scored_metadata.append((total_score, entry))

    scored_metadata.sort(reverse=True, key=lambda x: x[0])
    top_results = [entry for _, entry in scored_metadata[:3]]
    return top_results
