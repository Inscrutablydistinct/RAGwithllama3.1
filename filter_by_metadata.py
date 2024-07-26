import warnings
warnings.filterwarnings("ignore")
from model_param import embeddings
from sklearn.metrics.pairwise import cosine_similarity
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from model_param import CFG, embeddings
from langchain_community.vectorstores import FAISS
from datetime import datetime

def compute_cosine_similarity(text1, text2):
    embed1 = embeddings.embed_query(str(text1))
    return cosine_similarity([embed1], [text2])[0][0]

def filter_attributes(metadata_entry, key, value):
    if (key=='title'):
        cos_sim = compute_cosine_similarity(metadata_entry['title'], value)
        return cos_sim*5
    elif (key == 'author'):
        cos_sim = compute_cosine_similarity(metadata_entry['author'], value)
        return cos_sim*5
    elif (key == 'abstract'):
        cos_sim = compute_cosine_similarity(metadata_entry['abstract'], value)
        return cos_sim*5
    elif (key == 'keywords'):
        cos_sim = compute_cosine_similarity(metadata_entry['keywords'], value)
        return cos_sim*5
    elif (key == 'publication_date'):
        op = value[0] if value[1].isdigit() else value[0:2]
        value = value[len(op):]
        filter_date = datetime.strptime(value, "%Y-%m-%d")
        if metadata_entry['publication_date'] == "N/A":
            return 0.0
        entry_date = datetime.strptime(metadata_entry['publication_date'], "%Y-%m-%d")
        if (op == '>'):
            return 2.0 if entry_date > filter_date else -6.0
        elif (op == '>='):
            return 2.0 if entry_date >= filter_date else -6.0
        elif (op == '<'):
            return 2.0 if entry_date < filter_date else -6.0
        elif (op == '<='):
            return 2.0 if entry_date <= filter_date else -6.0
        else:
            return 2.0 if entry_date == filter_date else -6.0
    elif (key == 'results'):
        if (type(metadata_entry['results'])==list):
            metadata_entry['results'] = " ".join(metadata_entry['results'])
        cos_sim = compute_cosine_similarity(metadata_entry['results'], value)
        return cos_sim
    else:
        return 0.0

def filter_data(metadata, filter_dict):
    vectordb = FAISS.load_local(CFG.Output_folder + '/faiss_index_papers', # from output folder
        embeddings,
        allow_dangerous_deserialization = True,)
    scored_metadata = []
    store = {}
    for entry in metadata:
        total_score = 0.0
        for key, value in filter_dict.items():
            if (key == 'publication_date'):
                total_score += filter_attributes(entry, key, value)
            elif key in store:
                total_score += filter_attributes(entry, key, store[key])
            else:
                store[key] = embeddings.embed_query(value)
                total_score += filter_attributes(entry, key, store[key])
        print(total_score)
        scored_metadata.append((total_score, entry))

    scored_metadata.sort(reverse=True, key=lambda x: x[0])
    top_results = [entry for _, entry in scored_metadata[:3]]
    return top_results
