import warnings
warnings.filterwarnings("ignore")

import textwrap
import time

def wrap_text_preserve_newlines(text, width=1500):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


def process_llm_response(llm_response):
    if isinstance(llm_response, str):
        ans = wrap_text_preserve_newlines(llm_response)
    else:
        ans = wrap_text_preserve_newlines(llm_response['result'])
    pattern = "<|assistant|>"
    index = ans.find(pattern)
    if index != -1:
        ans = ans[index + len(pattern):]
    return ans.strip()

def llm_ans(llm_response):
    start = time.time()
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans + time_elapsed_str
