from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from langchain.schema import Document
from model_param import CFG

def extract_text(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

def text_split(md):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CFG.split_chunk_size,
        chunk_overlap = CFG.split_overlap
    )

    list_of_documents = []

    arr = [3948,4042,4111,486,504,3760,682,2271,429,748,1912,1665]
    j = 0
    for i in arr:
        file_path = f"{CFG.PDFs_path}/pdf{i}.pdf"
        text = extract_text(file_path)
        docs = text_splitter.split_documents([Document(page_content=text)])
        for chunk in docs:
            list_of_documents.append(Document(page_content=chunk.page_content, metadata=md[j]))
        j+=1
    return list_of_documents
