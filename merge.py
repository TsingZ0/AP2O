import argparse
import json
import numpy as np
import os
import pandas as pd

from utils.define import PASS_SCALAR


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )

    args = parser.parse_args()

    # Set the hyperparameters
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path


    # Load the samples
    for split in sorted(os.listdir(DATA_PATH)):

        results = None
        subfolder = os.path.join(DATA_PATH, split, 'runs')

        for file in os.listdir(subfolder):
            if file.endswith('.json'):
                dataset = pd.read_json(os.path.join(subfolder, file), orient='records', lines=True)
                results = dataset if results is None else pd.concat([results, dataset], ignore_index=True)

        # sort errors per problem
        error_frequency_per_task_id = results.groupby('task_id')['error_type'].value_counts(normalize=True).unstack(fill_value=0).to_dict(orient='index')
        
        # # sort errors in total
        # metrics = pd.read_json(os.path.join(subfolder, 'metrics.txt'), orient='records', lines=True)
        # error_counts = metrics['error_counts'].iloc[0]
        # total_errors = sum(error_counts.values())
        # error_frequency = {error_type: count / total_errors for error_type, count in error_counts.items()}

        results = results.groupby(['task_id', 'answer_id', 'prompt', 'answer', 'output', 'error_type'], as_index=False, sort=False)
        results = results[['avg_time', 'std_time']].mean()

        # Filter the samples
        total_prob = len(results['task_id'].unique())
        total_ans = len(results)

        # Score errors per problem
        results['error_frequency'] = results[['task_id', 'error_type']].apply(
            lambda x: PASS_SCALAR 
            if x['error_type'] == "" else error_frequency_per_task_id[x['task_id']].get(x['error_type'], np.inf),
            axis=1
        )

        # # Score errors in total
        # results['error_frequency'] = results[['task_id', 'error_type']].apply(
        #     lambda x: PASS_SCALAR 
        #     if x['error_type'] == "" else error_frequency.get(x['error_type'], np.inf),
        #     axis=1
        # )

        results['error_frequency'] = results['error_frequency'].fillna(value=np.inf)

        # # Score time
        # results['avg_time'] = pd.to_numeric(results['avg_time'], errors='coerce')
        # results['time_cost'] = (results['output'] == 'passed').astype(int) * results['avg_time']
        # results['time_cost'] = results['time_cost'].fillna(np.inf)

        # Score length
        results['answer_length'] = results['answer'].map(
            lambda x: len(x)
        )
        over_large = max(results['answer_length'])+10000
        results['answer_length'] = (results['output'] == 'passed').astype(int) * results['answer_length'] + \
        (results['output'] != 'passed').astype(int) * over_large

        results = results.sort_values([
            'task_id', 
            'error_frequency', 
            'answer_length'
        ])
        results = results.drop_duplicates(subset=['answer'])

        # filter the samples
        mask = results.groupby('task_id', sort=False)['error_frequency'].transform(
            lambda x : sum(x == PASS_SCALAR) >= 1 and sum(x != PASS_SCALAR) >= 2
        )
        results = results[mask]
        mask = results.groupby('task_id', sort=False)['answer_length'].transform(
            lambda x : sum(x != over_large) >= 2 and sum(x == over_large) >= 1
        )
        results = results[mask]

        assert len(results) > 0, f'No avaliable samples.'
        filtered_ans = len(results)
        filtered_prob = len(results['task_id'].unique())

        ratio_prob = round(filtered_prob / total_prob * 100, 2)
        ratio_ans = round(filtered_ans / total_ans * 100, 2)

        passed_count = results[results['error_frequency'] == PASS_SCALAR].groupby(
            'task_id', sort=False)['answer'].count()

        avg_time = results.groupby('task_id', sort=False)['avg_time'].mean()
        std_time = results.groupby('task_id', sort=False)['avg_time'].std()

        results = results.rename(columns={'task_id': 'id', 'prompt': 'problem', 'answer': 'text'})
        results['answers'] = results[['text', 'error_frequency', 'error_type', 'answer_length']].agg(
            lambda x: dict(zip(x.index, x.values)), axis=1)
        results = results[['id', 'problem', 'answers']]
        results = results.groupby(['id', 'problem'], sort=False)['answers'].apply(list).reset_index()

        results['count'] = passed_count.values
        results['avg'] = avg_time.values
        results['std'] = std_time.values
        results['cov'] = results['std'] / results['avg']

        results['time'] = results[['count', 'avg', 'std', 'cov']].agg(
            lambda x: dict(zip(x.index, x.values)), axis=1)
        results = results.drop(['count', 'avg', 'std', 'cov'], axis=1)


        # Save the dataset
        merge_path = os.path.join(SAVE_PATH, split)
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)
        
        results.to_json(os.path.join(merge_path, 'merged.json'), orient='records', lines=True)

        with open(os.path.join(merge_path, 'results.txt'), 'w') as f:
            print(f'Problems (total): {total_prob}', file=f)
            print(f'Problems (filtered): {filtered_prob}', file=f)
            print(f'Problems (ratio): {ratio_prob:.2f}\n', file=f)

            print(f'Answers (total): {total_ans}', file=f)
            print(f'Answers (filtered): {filtered_ans}', file=f)
            print(f'Answers (ratio): {ratio_ans:.2f}', file=f)

        print(f'\n{split.capitalize()}\n')
        with open(os.path.join(merge_path, 'results.txt'), 'r') as f:
            print(f.read())


if __name__ == '__main__':
    main()
