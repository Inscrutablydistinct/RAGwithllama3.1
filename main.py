from query_extraction import generate_md
import text_split
from model_param import CFG
from embeddings_and_context import make_context, make_embeddings
from langchain_community.vectorstores import FAISS
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from filter_by_metadata import filter_data
from data_preprocess import preprocess
import json
import time 
import warnings
import openai
from collections import defaultdict

warnings.filterwarnings("ignore")

with open('metadata.json') as f:
    d = json.load(f)
Question = """Your task is to identify the attributes/features of the metadata from a given user query. The attributes/features you need to identify are:

title
author
abstract
keywords
publication_date
arxiv_id
results

PLease note that:
- If a query specifies a date, include "<", ">", ">=", "<=", "=" to denote before, after, after and on, before and on, and on the publication date, respectively.
- If the query includes a metadata attribute term (e.g., author) without a specific value, do not identify it as a metadata attribute.
- The 'abstract' and 'keywords' attributes should always be identified, but they shouldn't be more than 10 words long.
- Always give the output in the form of a python dictionary. Only include the identified metadata attributes.

Examples:

1. Query: "I need the abstract and results from the recent paper on DNA bending after 27 August 2024.”
   Identified Attributes:
   abstract: 'paper on DNA bending'
   publication_date: '27 August 2024'
   keywords: 'DNA bending'
   Output: {"abstract": "paper on DNA bending", "publication_date": ">2024-08-27", "keywords": "DNA bending"}

2. Query: "Please provide the title and abstract of the latest research paper by Dr. Lee published on 15 June 2023 about AI in healthcare."
   Identified Attributes:
   author: 'Dr. Lee'
   abstract: 'study on healthcare'
   publication_date: '15 June 2023'
   keywords: 'AI in healthcare'
   Output: {"author": "Dr. Lee", "abstract": "study on healthcare", "publication_date": "=2023-06-15", "keywords": "AI in healthcare"}

3. Query: "Give me a novel way to devise therapeutic drugs to treat cancer?"
   Identified Attributes:
   keywords: 'cancer'
   abstract: 'A novel way to devise therapeutic drugs to treat cancer.'
   Output: {"abstract": "A novel way to devise therapeutic drugs to treat cancer.", "keywords":"cancer"}
The answer should only be a list and no other content whatsoever. Please print the Output for the following query:\n
"""
# Splitting documents
list_of_documents = text_split.text_split(d)
make_embeddings(list_of_documents)

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

# Function to generate answer
def ans(context, question):
    prompt = f"""
    You are given some extracted parts in a paragraph from research papers along with a question. Everything in the extract may not be important. Choose carefully!
    
    If you don't know the answer, just say "I don't know." Don't try to make up an answer.
    
    It is very important that you ALWAYS answer the question in the same language the question is in.
    
    Use the following pieces of context to answer the question at the end.
    
    Context: {context}
    
    Question is below.
    
    Question: {question}
    """

    response = client.chat.completions.create(
        model="phi3:latest",
        temperature=0.6,
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Use POST method to interact with this server.")

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))

            # Get the query from the JSON data
            query = request_data.get('messages', [None])[1].get('content')

            if query:
                start_time = time.time()

                out = generate_md(Question, query, client)
                filtered_metadata = filter_data(d, out)
                context = preprocess(make_context(list_of_documents, filtered_metadata[0], query))
                answer = ans(context, query)
                response_data = {
                    "answer": answer,
                    "source_document": filtered_metadata[0]['title'],
                    "time_taken": time.time() - start_time
                }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            else:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "No query provided"}).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            print(f"Error handling POST request: {str(e)}")

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('0.0.0.0', port) 
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()
if __name__ == "__main__":
    run()
