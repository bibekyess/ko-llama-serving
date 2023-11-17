from client_exllama import chat
import sys
sys.path.append('../../common/')
from questions import questions_korean
import pandas as pd

if __name__ == '__main__':
    counter = 1
    responses = []
    for q in questions_korean:
        response = chat(q)
        responses.append(response)

    df = pd.DataFrame(responses)
    df.to_csv('bench-exllama-gptq-4bit.csv', index=False)
