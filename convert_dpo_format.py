from datasets import Dataset
import pandas as pd
import os

DATA_PATH = "datasets/mbpp/Qwen2.5-Coder-3B-Instruct-0.6"

val_data = pd.read_json(os.path.join(DATA_PATH, 'validation', 'merged.json'), orient='records', lines=True)
val_data = Dataset.from_pandas(val_data)
print(len(val_data))

# ...existing code...

def dpo_format(example):
    """
    For one example, return a dictionary of lists for batched processing.
    """
    out_dict = {
        "id": [],
        "prompt": [], 
        "chosen": [],
        "rejected": [],
    }
    
    for idx in range(len(example["id"])):
        chosen_texts = [ans["text"] for ans in example["answers"][idx] if ans.get("error_type", "") == ""]
        rejected_texts = [ans["text"] for ans in example["answers"][idx] if ans.get("error_type", "") != ""]
        
        for c in chosen_texts:
            for r in rejected_texts:
                out_dict["id"].append(example["id"][idx])
                out_dict["prompt"].append(example["problem"][idx])
                out_dict["chosen"].append(c)
                out_dict["rejected"].append(r)
    
    return out_dict

# Keep batched=True
val_data = val_data.map(dpo_format, batched=True, remove_columns=val_data.column_names)
print(len(val_data))
for data in val_data:
    print(data.keys())
    input()