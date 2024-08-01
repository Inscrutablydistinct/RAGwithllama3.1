from model_param import CFG, embeddings
from langchain_community.vectorstores import FAISS
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def make_embeddings(list_of_documents):
    index_path = os.path.join(CFG.Embeddings_path, 'index.faiss')
    if not os.path.exists(index_path):
        print('Creating embeddings...\n\n')

        vectordb = FAISS.from_documents(
            documents=list_of_documents,
            embedding=embeddings
        )

        vectordb.save_local(f"{CFG.Output_folder}/faiss_index_papers")
        
def find_similar(list_of_documents, top):
    filtered_indices = []
    title = top['title']
    filtered_documents = [doc for doc in list_of_documents if doc.metadata.get("title") == title]
    for idx, doc in enumerate(list_of_documents):
        key = doc.metadata.get("title")
        if key == title:
            filtered_indices.append(idx)
    return filtered_indices, filtered_documents

def make_context(list_of_documents, top_md, query):
    vectordb = FAISS.load_local(CFG.Output_folder + '/faiss_index_papers',
        embeddings,
        allow_dangerous_deserialization = True,)
    filtered_indices, filtered_documents = find_similar(list_of_documents, top_md)
    if not filtered_indices:
        print("No documents found with the specified metadata.")
    else:
        filtered_embeddings = [vectordb.index.reconstruct(idx) for idx in filtered_indices]
        filtered_embeddings = np.array(filtered_embeddings)
        if len(filtered_embeddings.shape) == 1:
            filtered_embeddings = filtered_embeddings.reshape(1, -1)

        print(f"\n\n{query}\n\n")
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_embedding, filtered_embeddings).flatten()
        top_k_indices = similarities.argsort()[-4:][::-1]

        top_k_documents = [filtered_documents[i] for i in range(len(filtered_documents)) if i in top_k_indices]
        context = ""
        for doc in top_k_documents:
            context += " " + doc.page_content
    print(f"\n\n{context}\n\n")
    return remove_repeated_phrases(context)

def remove_repeated_phrases(text, chunk_size=400, overlap=0.2):
    """
    Remove repeated phrases from text.

    Parameters:
    - text: str, the input text to process
    - chunk_size: int, the size of chunks to compare for repetitions
    - overlap: float, fraction of overlap between chunks

    Returns:
    - str, text with repeated phrases removed
    """
    tokens = text.split()
    num_tokens = len(tokens)
    step_size = int(chunk_size * (1 - overlap))

    seen_chunks = set()
    cleaned_tokens = []

    for i in range(0, num_tokens, step_size):
        chunk = ' '.join(tokens[i:i + chunk_size])

        if chunk not in seen_chunks:
            cleaned_tokens.extend(tokens[i:i + chunk_size])
            seen_chunks.add(chunk)
        else:
            print(f"Skipped a repeated chunk: {chunk[:30]}...")

    return ' '.join(cleaned_tokens)
