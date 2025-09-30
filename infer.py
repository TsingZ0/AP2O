import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import deepspeed

import json
import pandas as pd
import torch

from datasets import load_from_disk
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils.evaluation import evaluate_functional_correctness
from utils.define import SYSTEM_PROMPT, EOS, post_process


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--eval_mode', type=str, choices=['True', 'False'], required=True, help='Whether to evaluate or sample.'
    )
    parser.add_argument(
        '--regen', type=str, choices=['True', 'False'], required=True, help='Whether to regenerate the outputs.'
    )
    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--model_path', type=str, help='The model path to use.'
    )
    parser.add_argument(
        '--n_seq', type=int, help='The number of independently computed returned sequences.'
    )
    parser.add_argument(
        '--n_iter', type=int, help='The number of iterations for sharding the inference.'
    )
    parser.add_argument(
        '--temp', type=float, help='The value used to modulate the next token probabilities.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='The seed to use.'
    )
    parser.add_argument(
        '--n_workers', type=int, default=1, help='The number of workers to use for evaluate_functional_correctness.'
    )
    parser.add_argument(
        '--n_check', type=int, help='The number of check for evaluate_functional_correctness.'
    )
    parser.add_argument(
        '--eval_k', type=str, default='', help='The number of check for evaluate_functional_correctness.'
    )
    parser.add_argument(
        '--look_output', type=bool, default=False, help='Whether to look at the output.'
    )

    args = parser.parse_args()
    print('== Inference Arguments ==')
    print(args)

    # Set the hyperparameters
    EVAL_MODE = args.eval_mode == 'True'
    REGEN = args.regen == 'True'
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    N_SEQ = args.n_seq
    N_ITER = args.n_iter
    TEMP = args.temp
    SAVE_PATH = args.save_path
    SEED = args.seed
    N_WORKERS = args.n_workers
    N_CHECK = args.n_check
    MAX_LEN = 1024
    if args.eval_k:
        EVAL_K = list(map(int, args.eval_k.split(',')))
        N_SEQ = EVAL_K[-1]
    else:
        EVAL_K = [1]


    # Set the seed
    set_seed(SEED)

    if REGEN:

        # Load the model
        subfolders = [f.path for f in os.scandir(MODEL_PATH) if f.is_dir() and 'checkpoint' in f.name]
        if subfolders:
            # Sort checkpoints by number
            subfolders.sort(key=lambda x: int(x.split('-')[-1]))
            print(f"All checkpoints: {subfolders}")
            MODEL_PATH = subfolders[-1]
            print(f"Using latest checkpoint from: {MODEL_PATH}")
        else:
            print(f'No checkpoint found in {MODEL_PATH}, using the original model.')

        sampling_params = SamplingParams(
            max_tokens=MAX_LEN,
            temperature=TEMP,
            n=N_SEQ,
            stop=EOS,
        )
        model = LLM(
            model=MODEL_PATH, 
            # gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
        )

    # Load the dataset
    data = load_from_disk(DATA_PATH)
    splits = ['test'] if EVAL_MODE else ['train', 'validation']

    for split in splits:
        problems = data[split].to_pandas()
        problems = problems.set_index('task_id', drop=False)
        problems = problems.to_dict('index')

        save_path = SAVE_PATH if EVAL_MODE else os.path.join(SAVE_PATH, split)
        print(f'Will save to {save_path}')

        if REGEN:
            try:
                os.makedirs(save_path)
            except:
                pass

            samples = []
            for problem in tqdm(problems.values()):
            # for idx, problem in tqdm(enumerate(problems.values())):
            #     if idx > 10:
            #         break

                with torch.no_grad():
                    prompt = SYSTEM_PROMPT.format(problem['prompt'].strip())

                    # Generate the samples
                    for _ in range(N_ITER):
                        outputs = model.generate(prompt, sampling_params)
                        outputs = [output.text for output in outputs[0].outputs]

                        for answer in outputs:
                            # print('+'*50, prompt)
                            # print('+'*50, answer)
                            # input()
                            answer = post_process(answer)
                            # answer = post_process(answer, replace_entrance_name=True)
                            # print('+'*50, answer)
                            # input()

                            samples.append(dict(task_id=problem['task_id'], answer=answer))

            with open(os.path.join(save_path, 'outputs.json'), 'w') as f:
                json.dump(samples, f)
                
            print(f'{split}: {len(samples)} samples generated')

        else:
            with open(os.path.join(save_path, 'outputs.json'), 'r') as f:
                samples = json.load(f)

            subfolder = os.path.join(save_path, 'runs')
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder)

            for i in range(N_CHECK):
                print(f'\nCheck {i}:\n')

                file = os.path.join(subfolder, f'check_{i}.json')
                # if os.path.isfile(file):
                #     print('Skipping as results already exists.')
                #     continue

                results = evaluate_functional_correctness(samples, problems, subfolder, k=EVAL_K, n_workers=N_WORKERS)
                results = pd.DataFrame([r[1] for result in results.values() for r in result])
                results.to_json(file, orient='records', lines=True)
                print(f'Results saved to {file}')

    print('Infer done!')


if __name__ == '__main__':
    main()
    time.sleep(60)
