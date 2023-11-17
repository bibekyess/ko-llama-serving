# Env variable is openllm-env
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys
sys.path.append('../common/')
from questions import questions_korean
import pandas as pd


from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig
from peft import PeftModel
import torch

model_id = "daekeun-ml/Llama-2-ko-instruct-13B"
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=nf4_config)
model = PeftModel.from_pretrained(model, "neoALI/table-peft")
model.to("cuda")


def predict(prompt:str):
    start_time = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=400,
                                   num_beams=3,
                                   length_penalty = 1.1
                                   )
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    request_time = time.perf_counter() - start_time
    tok_count = len(tokenizer.encode(output))
    tokens_per_second = tok_count/request_time

    return {
        'Framework': 'hf-bnb-nf4bit',
        'Generated_tokens': len(tokenizer.encode(output)),
        'Inference_time': format(request_time, '.2f'),
        'Tokens/sec': format(tokens_per_second, '.2f'),
        'Question': prompt,
        'Answer': output,
        }



if __name__ == '__main__':
    responses = []

    for q in questions_korean:
        response = predict(q)
        print(response)
        responses.append(response)

    df = pd.DataFrame(responses)
    df.to_csv('bench_hf-bnb-nf4bit.csv', index=False)

# 12.3GB