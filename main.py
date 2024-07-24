from query_extraction import generate_md
import text_split
from model_param import CFG
from embeddings_and_context import make_context
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

Note:
- The attribute 'keywords' must be non-empty and always present in the final output.
- The main query should be the input user query.
- If a query specifies a date, include "<", ">", ">=", "<=", "=" to denote before, after, after and on, before and on, and on the publication date, respectively.
- From the user query make the main query and the metadata attributes.
- If the query includes a metadata attribute term (e.g., author) without a specific name, include it in the main query instead of identifying it as a metadata attribute.
- The 'abstract' attribute should be present in all, but it shouldn't be more than 20 words long.
- Give the output at all costs, always include the main query and the non empty/null identified metadata attributes.

Examples:
1. Query: "Can you tell me the authors of the paper titled 'An Alternative Source for Dark Energy'?”
   Identified Attributes:
   title: 'An Alternative Source for Dark Energy'
   abstract: 'An Alternative Source for Dark Energy' 
   keywords: 'Alternative Source, Dark Energy'
   Main Query: "Can you tell me the authors of the paper."
   Output: ["Can you tell me the authors of the paper titled 'An Alternative Source for Dark Energy'", {"title": "An Alternative Source for Dark Energy","abstract": "An Alternative Source for Dark Energy" ,"keywords": "Alternative Source, Dark Energy"}]

2. Query: "I need the abstract and results from the recent paper on DNA bending after 27 August 2024.”
   Identified Attributes:
   abstract: 'paper on DNA bending'
   publication_date: '27 August 2024'
   keywords: 'DNA bending'
   Main Query: "I need the abstract and results from the recent paper."
   Output: ["I need the abstract and results from the recent paper on DNA bending after 27 August 2024.", {"abstract": "paper on DNA bending", "publication_date": ">2024-08-27", "keywords": "DNA bending"}]

3. Query: "I want the abstract of the research paper on Chain Theory written by Dr. Mazur.”
   Identified Attributes:
   title: 'Chain Theory'
   author: 'Dr. Mazur'
   abstract: 'paper on Chain Theory'
   keywords: 'Chain Theory'
   Main Query: "I want the abstract of the research paper."
   Output: ["I want the abstract of the research paper on Chain Theory written by Dr. Mazur.", {"title": "Chain Theory", "author": "Dr. Mazur", "abstract": "paper on Chain Theory","keywords": "Chain Theory"}]

4. Query: "Please provide the title and abstract of the latest research paper by Dr. Lee published on 15 June 2023 about AI in healthcare."
   Identified Attributes:
   author: 'Dr. Lee'
   abstract: 'study on healthcare'
   publication_date: '15 June 2023'
   keywords: 'AI in healthcare'
   Main Query: "Please provide the title and abstract of the latest research paper."
   Output: ["Please provide the title and abstract of the latest research paper about AI in healthcare.", {"author": "Dr. Lee", "abstract": "study on healthcare", "publication_date": "=2023-06-15", "keywords": "AI in healthcare"}]

5. Query: "Give me a novel way to devise therapeutic drugs to treat cancer?"
   Identified Attributes:
   keywords: 'cancer'
   abstract: 'A novel way to devise therapeutic drugs to treat cancer.'
   Main Query: "Give me a novel way to devise therapeutic drugs to treat cancer."
   Output: ["Give me a novel way to devise therapeutic drugs to treat cancer.", {"abstract": "A novel way to devise therapeutic drugs to treat cancer.", "keywords":"cancer"}]

The answer should only be a list and no other content whatsoever. Please print the Output for the following query:\n
"""
# Splitting documents
list_of_documents = text_split.text_split(d)

# Initializing the OpenAI client
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

# Function to generate answer
def ans(context, question):
    prompt = f"""
    You are given some extracted parts in a paragraph from research papers along with a question. Everything in the extract may not be important. Choose carefully!
    
    If you don't know the answer, just say "I don't know." Don't try to make up an answer.
    
    It is very important that you ALWAYS answer the question in the same language the question is in. Remember to always do that.
    
    Use the following pieces of context to answer the question at the end.
    
    Context: {context}
    
    Question is below. Remember to answer only in English:
    
    Question: {question}
    """

    response = client.chat.completions.create(
        model="llama3.1:8b",
        temperature=0.5,
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
                filtered_metadata = filter_data(d, out[1])
                context = preprocess(make_context(list_of_documents, filtered_metadata[0], out))
                answer = ans(context, out[0])
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
