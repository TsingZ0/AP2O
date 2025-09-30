from datasets import load_dataset

# Path to your JSONL file
jsonl_path = "datasets/HumanEval/data/humaneval-python.jsonl"

# Load as a DatasetDict with a “test” split
dataset = load_dataset("json", data_files={"test": jsonl_path})

save_dir = "datasets/humanEval"
dataset.save_to_disk(save_dir)