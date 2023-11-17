import pandas as pd

# List of file paths to the CSV files you want to merge
directory_paths = ['/home/bibekyess/huggingface_transformer/bench_hf-bnb-nf4bit.csv', 
                   '/home/bibekyess/exllama_sandbox/exllama/bench-exllama-gptq-4bit.csv',
                   '/home/bibekyess/vllm_sandbox/bench-vllm-awq-4bit.csv', 
                   '/home/bibekyess/vllm_sandbox/bench-vllm-16bit.csv'
                   ]

# Initialize an empty list to store DataFrames read from CSV files
dataframes = []

# Loop through each file path, read the CSV file, and store the DataFrame in the list
for file in directory_paths:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
merged_data = pd.concat(dataframes, ignore_index=True)

# print(merged_data)
# Write the merged data to a new CSV file
merged_data.to_csv('/home/bibekyess/openllm_sandbox/benchmark_results.csv', index=False)
