from process_output import process_llm_response
import re
import ast
import subprocess
import json

with open('metadata.json') as f:
    d = json.load(f)

def generate_md(Question, query, client):
    prompt = f"{Question}{query}"
    response = client.chat.completions.create(
      model="phi3:latest",
      temperature=0.2,
      n=1,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],)
    text = response.choices[0].message.content

    # if result.returncode != 0:
    #     print(f"Error running curl: {result.stderr}")  # Debugging statement
    #     return "[]"
    
    # try:
    #     response = json.loads(result.stdout)
    #     text = response['choices'][0]['message']['content']
    text = process_llm_response(text)
    print(f"LLM Response: {text}")  # Debugging statement
    
    pattern = r'\{(?:\s*\"[^\"]*\"\s*:\s*\"[^\"]*\"\s*,?)*\}'
    match = re.search(pattern, text)
    if match:
        output_list = match.group(0)
        return ast.literal_eval(output_list)
    else:
        print("No match found")
        return "[]"
