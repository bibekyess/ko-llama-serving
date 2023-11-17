import requests, json, time
from transformers import AutoTokenizer

model_id = "daekeun-ml/Llama-2-ko-instruct-13B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def chat(prompt:str):
    payload = {"prompt": prompt}
    settings = {"max_new_tokens": 400, "beams": 3, "beam_length": 3, "in_beam_search": True}
    payload["settings"] = settings
    headers = {'Content-Type': 'application/json'}
    
    start = time.perf_counter()
    
    response = requests.post("http://localhost:8004/infer_bench", headers=headers, data=json.dumps(payload))
    # response = requests.post("http://localhost:8004/infer_bench", data=payload)
    # print(response.text)
    generated_text = response.text
    print(generated_text)
    request_time = time.perf_counter() - start
    tok_count = len(tokenizer.encode(generated_text))
    tokens_per_second = tok_count/request_time

    return {
        'Framework': 'exllama-gptq-4bit',
        'Generated_tokens': len(tokenizer.encode(generated_text)),
        'Inference_time': format(request_time, '.2f'),
        'Tokens/sec': format(tokens_per_second, '.2f'),
        'Question': prompt,
        'Answer': generated_text,
        }

if __name__ == '__main__':
    prompt = "Who are you?"
    print(f"User: {prompt}\nLlama2: {chat(prompt)})")