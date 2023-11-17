import requests, json, time
from transformers import AutoTokenizer

model_id = "daekeun-ml/Llama-2-ko-instruct-13B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def chat(prompt:str):
    payload = {"prompt": prompt, "max_tokens": 400, 
                "use_beam_search": True,
                "length_penalty": 1.1,
                "best_of": 3,
                "n": 1,
                "temperature": 0
                }
    headers = {'Content-Type': 'application/json'}
    start = time.perf_counter()
    
    response = requests.post("http://localhost:8000/generate", headers=headers, data=json.dumps(payload))
    generated_text = response.json()["text"][0]
    
    request_time = time.perf_counter() - start
    tok_count = len(tokenizer.encode(generated_text))
    tokens_per_second = tok_count/request_time

    return {
        'Framework': 'vllm-awq-4bit',
        # 'Framework': ' vllm-16bit',        
        'Generated_tokens': len(tokenizer.encode(generated_text)),
        'Inference_time': format(request_time, '.2f'),
        'Tokens/sec': format(tokens_per_second, '.2f'),
        'Question': prompt,
        'Answer': generated_text,
        }

if __name__ == '__main__':
    prompt = "Explain about the relationship between Nepal and Korea"
    print(f"User: {prompt}\nLlama2: {chat(prompt)['Answer']})")