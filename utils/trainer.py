
import re
import warnings
import datasets
import torch
import numpy as np
import torch.amp as amp

from tqdm import tqdm
from trl import SFTTrainer, DPOTrainer
from trl.trainer.utils import pad_to_length
from torch.utils.data import DataLoader
from typing import Optional
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging, is_torch_xpu_available
from collections import defaultdict
from collections import Counter
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk
from contextlib import nullcontext

from .evaluation import estimate_pass_at_k, check_correctness
from .define import post_process, EOS


logger = logging.get_logger(__name__)


class MySFTTrainer(SFTTrainer):

    def _prepare_non_packed_dataloader(
        self,
        processing_class,
        dataset,
        dataset_text_field: str,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = processing_class(
                element[dataset_text_field] if formatting_func is None else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if formatting_func is not None and not isinstance(formatting_func(element), list):
                raise ValueError(
                    "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                )

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        if dataset.column_names is not None:  # None for IterableDataset
            extra_columns = list(set(dataset.column_names) - set(signature_columns))
        else:
            extra_columns = []

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with "
                "the default collator and yield to errors. If you want to inspect dataset other columns (in this "
                f"case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the "
                "default collator and create your own data collator in order to inspect the unused dataset columns.",
                UserWarning,
            )

        map_kwargs = {
            "batched": False,
            "remove_columns": dataset.column_names if remove_unused_columns else None,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)

        return tokenized_dataset
    

def run_code_unit_test(k, samples, problems, metric_key_prefix, n_workers=1, timeout=3.0):
    # Check the generated samples against test suites
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        answer_id = Counter()
        n_samples = 0

        for sample in samples:
            task_id = sample['task_id']
            answer = sample['answer']
            any_key = list(problems.keys())[0]
            if type(task_id) is not type(any_key):
                task_id = type(any_key)(task_id)
            
            args = (problems[task_id], answer, timeout, answer_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            
            answer_id[task_id] += 1
            n_samples += 1

        results = defaultdict(list)

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result['task_id']].append((result['answer_id'], result))

    # Calculate metrics
    total, correct, speed, error_types, answer_lengths = [], [], [], [], []

    for result in results.values():

        passed = []
        result.sort()

        for r in result:
            passed.append(r[1]['output'] == 'passed')
            if passed[-1] == True:
                speed.append(r[1]['avg_time'])
                answer_lengths.append(r[1]['answer_length'])
            error_types.append(r[1]['error_type'])

        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)
    speed = np.array(speed)
    error_counts = Counter(error_types)

    metrics = {}
    metrics['avg_answer_length'] = np.mean(answer_lengths)
    metrics['avg_time'] = np.mean(speed)
    metrics['error_counts'] = dict(sorted(error_counts.items(), key=lambda item: -item[1]))
    metrics[f"{metric_key_prefix}_pass@{k}"] = estimate_pass_at_k(total, correct, k)

    return metrics
    

class MyDPOTrainer(DPOTrainer):

    def __init__(self, *args, **kwargs):
        if 'data_collator' in kwargs:
            kwargs['data_collator'].trainer = self

        self.EVAL_MAX_LEN = kwargs['EVAL_MAX_LEN']
        self.EVAL_BATCH_SIZE = kwargs['EVAL_BATCH_SIZE']
        self.SAMPLE = kwargs['SAMPLE']
        self.TEMP = kwargs['TEMP']
        self.TRAIN_DATA_PATH = kwargs['TRAIN_DATA_PATH']
        self.val_metrics = {}
        for key in ['EVAL_MAX_LEN', 'EVAL_BATCH_SIZE', 'SAMPLE', 'TEMP', 'TRAIN_DATA_PATH']:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)
    
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        if self.generate_during_eval and self.args.metric_for_best_model.startswith('pass'):
            k = 1
            # TODO large k
            # match = re.search(r'@(\d+)', self.args.metric_for_best_model)
            # k = int(match.group(1))

            data = load_from_disk(self.TRAIN_DATA_PATH)
            problems = data['validation'].to_pandas()
            problems = problems.set_index('task_id', drop=False)
            problems = problems.to_dict('index')

            samples = []
            num_samples = len(dataloader.dataset)
            step_size = self.EVAL_BATCH_SIZE
            for start_index in tqdm(range(0, num_samples, step_size)):
                end_index = min(start_index + step_size, num_samples)
                indices = list(range(start_index, end_index))
                if len(indices) < 2:
                    break

                batch_dataset = dataloader.dataset.select(indices)
                batch = self.data_collator(batch_dataset)
                batch = self._prepare_inputs(batch)

                policy_output_decoded = self.generate_from_model(k, self.model, batch)
                for data, response in zip(batch_dataset, policy_output_decoded):
                    response = response[len(data['problem']):]
                    answer = post_process(response)
                    samples.append(dict(task_id=data['id'], answer=answer))

            new_metrics = run_code_unit_test(k, samples, problems, metric_key_prefix)

        # Base evaluation
        initial_output = super(DPOTrainer, self).evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        if self.generate_during_eval and self.args.metric_for_best_model.startswith('pass'):
            initial_output.metrics.update(new_metrics)
            self.val_metrics = initial_output.metrics

        return initial_output


    def generate_from_model(self, k, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        generate_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=self.EVAL_MAX_LEN,
                do_sample=self.SAMPLE,
                temperature=self.TEMP,
                num_return_sequences=k,
                pad_token_id=self.processing_class.pad_token_id,
                eos_token_id=self.processing_class.eos_token_id,
                stop_strings=EOS,
                tokenizer=self.processing_class,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.padding_value)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded
    
    