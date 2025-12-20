# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
from trl import DataCollatorForCompletionOnlyLM
from trl.trainer.dpo_trainer import DataCollatorForPreference
from typing import Any, Dict, List, Union, Optional
from utils.define import SYSTEM_PROMPT, PASS_SCALAR


def add_sys_prompt(example):
    example['problem'] = SYSTEM_PROMPT.format(example['problem'])
    return example


def sft_format(example):
    
    output_texts = []
    for a in example['answers']:
        prompt = example['problem']
        answer = a['text']
        text = f"### Question: {prompt}\n ### Answer: {answer}"
        output_texts.append(text)

    return output_texts


def select_pass(example):

    answers = [a for a in example['answers'] if a['error_frequency'] == PASS_SCALAR]
    example['answers'] = answers

    return example


class MySFTCollator(DataCollatorForCompletionOnlyLM):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:

        tokenized_batch = []
        for example in examples:

            i = random.choice(range(len(example['input_ids'])))
            batch_element = {'input_ids': example['input_ids'][i], 'attention_mask': example['attention_mask'][i]}
            tokenized_batch.append(batch_element)

        return super().torch_call(tokenized_batch)



def dpo_format(example):
    out_dict = {
        "id": [],
        "problem": [], 
        "prompt": [], 
        "chosen": [],
        "rejected": [],
    }
    
    for idx in range(len(example["id"])):
        chosen_texts = [a["text"] for a in example["answers"][idx] if a.get("error_type", "") == ""]
        rejected_texts = [a["text"] for a in example["answers"][idx] if a.get("error_type", "") != ""]
        
        for c in chosen_texts:
            for r in rejected_texts:
                out_dict["id"].append(example["id"][idx])
                out_dict["problem"].append(example['problem'][idx])
                out_dict["prompt"].append(example['problem'][idx])
                out_dict["chosen"].append(c)
                out_dict["rejected"].append(r)
    
    return out_dict


def curri_dpo_format(example, frequency_increase=True):
    out_dict = {
        "id": [],
        "problem": [], 
        "prompt": [], 
        "chosen": [],
        "rejected": [],
    }
    
    for idx in range(len(example["id"])):
        chosen_texts = [(a["text"], float(a['error_frequency'])) for a in example["answers"][idx] if a.get("error_type", "") == ""]
        rejected_texts = [(a["text"], float(a['error_frequency'])) for a in example["answers"][idx] if a.get("error_type", "") != ""]
        
        if frequency_increase:
            chosen_texts = sorted(chosen_texts, key=lambda x: x[1])
            rejected_texts = sorted(rejected_texts, key=lambda x: x[1])
        else:
            chosen_texts = sorted(chosen_texts, key=lambda x: x[1], reverse=True)
            rejected_texts = sorted(rejected_texts, key=lambda x: x[1], reverse=True)

        for c in chosen_texts:
            for r in rejected_texts:
                out_dict["id"].append(example["id"][idx])
                out_dict["problem"].append(example['problem'][idx])
                out_dict["prompt"].append(example['problem'][idx])
                out_dict["chosen"].append(c[0])
                out_dict["rejected"].append(r[0])
    
    return out_dict


def dpo_one_pair_format(example):
    out_dict = {
        "id": [],
        "problem": [], 
        "prompt": [], 
        "chosen": [],
        "rejected": [],
    }
    
    for idx in range(len(example["id"])):
        chosen_texts = [a["text"] for a in example["answers"][idx] if a.get("error_type", "") == ""]
        rejected_texts = [a["text"] for a in example["answers"][idx] if a.get("error_type", "") != ""]
        
        out_dict["id"].append(example["id"][idx])
        out_dict["problem"].append(example['problem'][idx])
        out_dict["prompt"].append(example['problem'][idx])
        out_dict["chosen"].append(random.choice(chosen_texts))
        out_dict["rejected"].append(random.choice(rejected_texts))
    
    return out_dict


def psudo_format(example, column_name):
    
    return {column_name: ''}


def dpo_dyn_format(example, task, trainer=None):
    if trainer is not None:
        current_epoch = trainer.state.epoch
        total_epochs = trainer.args.num_train_epochs
        val_metrics = trainer.val_metrics

    responses, scores, types = [], [], []
    for a in example['answers']:
        assert type(a['text']) == str, f"answer should be str, but got {type(a['text'])}"
        assert type(a['error_frequency']) in [int, float], f"error_frequency should be int or float, but got {type(a['error_frequency'])}"
        
        responses.append(a['text'])
        scores.append(float(a['error_frequency']))
        types.append(a['error_type'])

    responses, scores, types = np.array(responses), np.array(scores), np.array(types)
    mask = scores == PASS_SCALAR
    type_names, indices = np.unique(types, return_index=True)
    type_names = type_names[np.argsort(indices)] # keep order
    error_type_names = type_names[type_names != '']

    if task == 'pvf':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]
        rejected_response = responses[random.choice(np.flatnonzero(~mask))]

    elif task == 'pvf_iter':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)

        current_rejected_response_range = rejected_response_inds[start_idx:end_idx]
        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_type':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        total_elements = len(error_type_names)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)

        current_error_types = error_type_names[start_idx:end_idx]
        current_rejected_response_range = []
        for error_type in current_error_types:
            error_type_mask = types == error_type
            current_rejected_response_range.extend(np.flatnonzero(error_type_mask))

        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_expand':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)

        current_rejected_response_range = rejected_response_inds[0:end_idx]
        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_priority':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)

        current_rejected_response_range = rejected_response_inds[start_idx:end_idx]

        if 'error_counts' in val_metrics:
            current_rejected_len = len(current_rejected_response_range)

            added_rejected_response_inds = []
            val_total_errors = sum([
                error_cnt 
                    for error_type, error_cnt in val_metrics['error_counts'].items()
                        if error_type != ''
            ])
            all_error_cnts = []
            for error_type, error_cnt in val_metrics['error_counts'].items():
                if error_type != '':
                    all_error_cnts.append(error_cnt)

            if len(all_error_cnts) == 0:
                print("No error counts found in the current validation metrics. Replay is not activated.")
            else:
                least_error_cnt = min(all_error_cnts)
                val_total_errors = sum(all_error_cnts)

                least_error_cnts = {}
                for error_type, error_cnt in val_metrics['error_counts'].items():
                    if error_type != '':
                        least_error_cnts[error_type] = int(np.ceil(error_cnt / least_error_cnt))
                        
                for error_type, error_cnt in val_metrics['error_counts'].items():
                    if error_type != '':
                        type_mask = types == error_type
                        type_response_inds = np.flatnonzero(type_mask)

                        if len(type_response_inds) > 0:
                            error_ratio = error_cnt / val_total_errors
                            added_rejected_cnt = int(current_rejected_len * error_ratio)
                            added_rejected_cnt = max(least_error_cnts[error_type], added_rejected_cnt)

                            added_rejected_response_inds.extend(np.random.choice(
                                type_response_inds, size=added_rejected_cnt, replace=True).tolist())

                added_rejected_response_inds = np.array(added_rejected_response_inds, dtype=np.int64)
                current_rejected_response_range = np.append(
                    current_rejected_response_range,
                    added_rejected_response_inds
                )

        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_reverse':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)
        reverse_start_idx = total_elements - end_idx
        reverse_end_idx = total_elements - start_idx

        current_rejected_response_range = rejected_response_inds[reverse_start_idx:reverse_end_idx]
        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_reverse_priority':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor(current_epoch * elements_per_epoch))
        end_idx = int(np.floor((current_epoch + 1) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)
        reverse_start_idx = total_elements - end_idx
        reverse_end_idx = total_elements - start_idx

        current_rejected_response_range = rejected_response_inds[reverse_start_idx:reverse_end_idx]

        if 'error_counts' in val_metrics:
            current_rejected_len = len(current_rejected_response_range)

            added_rejected_response_inds = []
            val_total_errors = sum([
                error_cnt 
                    for error_type, error_cnt in val_metrics['error_counts'].items()
                        if error_type != ''
            ])
            all_error_cnts = []
            for error_type, error_cnt in val_metrics['error_counts'].items():
                if error_type != '':
                    all_error_cnts.append(error_cnt)

            if len(all_error_cnts) == 0:
                print("No error counts found in the current validation metrics. Replay is not activated.")
            else:
                least_error_cnt = min(all_error_cnts)
                val_total_errors = sum(all_error_cnts)

                least_error_cnts = {}
                for error_type, error_cnt in val_metrics['error_counts'].items():
                    if error_type != '':
                        least_error_cnts[error_type] = int(np.ceil(error_cnt / least_error_cnt))
                        
                for error_type, error_cnt in val_metrics['error_counts'].items():
                    if error_type != '':
                        type_mask = types == error_type
                        type_response_inds = np.flatnonzero(type_mask)

                        if len(type_response_inds) > 0:
                            error_ratio = error_cnt / val_total_errors
                            added_rejected_cnt = int(current_rejected_len * error_ratio)
                            added_rejected_cnt = max(least_error_cnts[error_type], added_rejected_cnt)

                            added_rejected_response_inds.extend(np.random.choice(
                                type_response_inds, size=added_rejected_cnt, replace=True).tolist())

                added_rejected_response_inds = np.array(added_rejected_response_inds, dtype=np.int64)
                current_rejected_response_range = np.append(
                    current_rejected_response_range,
                    added_rejected_response_inds
                )

        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_type_reverse':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        total_elements = len(error_type_names)
        elements_per_epoch = total_elements / total_epochs
        
        start_idx = min(
            int(np.floor((total_epochs - current_epoch - 1) * elements_per_epoch)),
            total_elements - 1
        )
        end_idx = min(
            int(np.floor((total_epochs - current_epoch) * elements_per_epoch)),
            total_elements
        )
        
        # Ensure we always have at least one element in range
        start_idx = min(start_idx, total_elements - 1)
        end_idx = max(start_idx + 1, end_idx)

        current_error_types = error_type_names[start_idx:end_idx]
        current_rejected_response_range = []
        for error_type in current_error_types:
            error_type_mask = types == error_type
            current_rejected_response_range.extend(np.flatnonzero(error_type_mask))

        rejected_response = responses[random.choice(current_rejected_response_range)]

    elif task == 'pvf_iter_reverse_expand':

        chosen_response = responses[random.choice(np.flatnonzero(mask))]

        rejected_response_inds = np.flatnonzero(~mask)

        total_elements = len(rejected_response_inds)
        elements_per_epoch = total_elements / total_epochs
        start_idx = int(np.floor((total_epochs - current_epoch - 1) * elements_per_epoch))
        end_idx = int(np.floor((total_epochs - current_epoch) * elements_per_epoch))
        # If end_idx equals start_idx, adjust end_idx to include at least one element
        if end_idx == start_idx:
            end_idx = min(start_idx + 1, total_elements)

        current_rejected_response_range = rejected_response_inds[start_idx:total_elements]
        rejected_response = responses[random.choice(current_rejected_response_range)]

    return {'chosen_response': chosen_response, 'rejected_response': rejected_response}
    

class MyDPOCollator(DataCollatorForPreference):

    def __init__(self, tokenizer, pad_token_id, kwargs=None):
        super().__init__(pad_token_id)
        self.tokenizer = tokenizer
        self.task = None
        self.trainer = None
        self.max_prompt_length = tokenizer.model_max_length
        self.max_completion_length = tokenizer.model_max_length // 2

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:

        token_examples = []
        for example in examples:

            output = dpo_dyn_format(
                example, 
                self.task, 
                self.trainer
            ) if 'answers' in example else example
            chosen_response = output['chosen_response']
            rejected_response = output['rejected_response']

            prompt_input_ids = self.tokenizer(example['problem'], add_special_tokens=False)["input_ids"]
            chosen_input_ids = self.tokenizer(chosen_response, add_special_tokens=False)["input_ids"]
            rejected_input_ids = self.tokenizer(rejected_response, add_special_tokens=False)["input_ids"]

            chosen_input_ids = chosen_input_ids + [self.tokenizer.eos_token_id]
            rejected_input_ids = rejected_input_ids + [self.tokenizer.eos_token_id]

            # Truncate prompt and completion sequences
            if self.max_prompt_length is not None:
                prompt_input_ids = prompt_input_ids[-self.max_prompt_length:]
            if self.max_completion_length is not None:
                chosen_input_ids = chosen_input_ids[:self.max_completion_length]
                rejected_input_ids = rejected_input_ids[:self.max_completion_length]

            token_example = {
                "prompt_input_ids": prompt_input_ids,
                "chosen_input_ids": chosen_input_ids,
                "rejected_input_ids": rejected_input_ids
            }

            token_examples.append(token_example)

        return super().torch_call(token_examples)
