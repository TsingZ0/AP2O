import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
import warnings

from datasets import disable_caching
disable_caching()

from datasets import Dataset
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.utils import logging
from trl import SFTConfig, DPOConfig

from utils.collator import *
from utils.trainer import MySFTTrainer, MyDPOTrainer

logging.set_verbosity_info()
logger = logging.get_logger('transformers')
logger.info('INFO')
logger.warning('WARN')


def main():

    # Define the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path', type=str, required=True, help='The data path to use.'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='The model path to use.'
    )
    parser.add_argument(
        '--optim', type=str, help='The optimization to use.'
    )
    parser.add_argument(
        '--task', type=str, help='The pairwise task for DPO.'
    )
    parser.add_argument(
        '--ds_config', type=str, required=True, help='The deepspeed config to use.'
    )
    parser.add_argument(
        '--save_path', type=str, required=True, help='The save path to use.'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='The seed to use.'
    )
    parser.add_argument(
        '--batch_size', type=int, required=True, help='The batch size to use.'
    )
    parser.add_argument(
        '--num_epochs', type=int, required=True, help='The number of epochs to use.'
    )
    parser.add_argument(
        '--temp', type=float, help='The value used to modulate the next token probabilities.'
    )
    parser.add_argument(
        '--train_data_path', type=str, required=True, help='The data path to use.'
    )

    args = parser.parse_args()
    print('== Training Arguments ==')
    print(args)

    # Set the hyperparameters
    DATA_PATH = args.data_path
    MODEL_PATH = args.model_path
    OPTIM = args.optim
    TASK = args.task
    DS_CONFIG = args.ds_config
    SAVE_PATH = args.save_path
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    MAX_LEN = 2048
    EVAL_MAX_LEN = 1024
    EVAL_BATCH_SIZE = 8
    NUM_EPOCHS = args.num_epochs
    SAMPLE = args.temp > 0.0
    TEMP = args.temp
    TRAIN_DATA_PATH = args.train_data_path

    print(f'Will save to {SAVE_PATH}')


    # Define the arguments
    train_args = TrainingArguments(
        output_dir=SAVE_PATH,
        eval_strategy='epoch',
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-7,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        logging_strategy='epoch',
        num_train_epochs=NUM_EPOCHS,
        save_strategy='best',
        save_total_limit=1,
        save_only_model=True,
        # load_best_model_at_end=True, # will store the optimizer, scheduler & rng state in checkpoint/global_step
        # fp16=True,
        bf16=True,
        remove_unused_columns=False,
        deepspeed=DS_CONFIG,
        report_to=['tensorboard'],
        gradient_checkpointing=False,
        dataloader_drop_last=False,
    )

    # Set the seed
    set_seed(SEED)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        model_max_length=MAX_LEN,
        use_fast=True,
    )
    print(tokenizer)
    # special_tokens = tokenizer.additional_special_tokens
    # added_tokens = [token.content for token in tokenizer.added_tokens_decoder.values()]
    # special_tokens = set(special_tokens + added_tokens).to_list()
    # print(f"Tokenizer: {special_tokens}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"Tokenizer existing pad token: {tokenizer.pad_token}")
    print(f"Tokenizer existing eos token: {tokenizer.eos_token}")
    tokenizer.padding_side = 'right'

    # Load the dataset
    train_data = pd.read_json(os.path.join(DATA_PATH, 'train', 'merged.json'), orient='records', lines=True)
    train_data = Dataset.from_pandas(train_data)
    train_data = train_data.map(add_sys_prompt)

    val_data = pd.read_json(os.path.join(DATA_PATH, 'validation', 'merged.json'), orient='records', lines=True)
    val_data = Dataset.from_pandas(val_data)
    val_data = val_data.map(add_sys_prompt)

    if OPTIM == 'sft':
        train_data = train_data.map(select_pass)
        val_data = val_data.map(select_pass)

        response_template = tokenizer(' ### Answer:', add_special_tokens=False)['input_ids']
        if model.config.model_type == 'llama':
            response_template = response_template[1:]

        data_collator = MySFTCollator(response_template, tokenizer=tokenizer)

        data_module = dict(
            data_collator=data_collator,
            train_dataset=train_data, 
            eval_dataset=val_data,
        )

        data_module['formatting_func'] = sft_format
        train_args.prediction_loss_only = True
        train_args.metric_for_best_model = 'eval_loss'
        train_args.greater_is_better = False

        args_as_dict = train_args.to_dict()
        train_args = SFTConfig(**args_as_dict)
        trainer = MySFTTrainer(
            model=model, 
            processing_class=tokenizer, 
            args=train_args, 
            **data_module, 
        )

    elif OPTIM == 'dpo':
        if TASK == 'passk':
            train_data = train_data.map(dpo_format, batched=True, remove_columns=val_data.column_names)
        elif TASK == 'curri':
            train_data = train_data.map(curri_dpo_format, batched=True, remove_columns=val_data.column_names)
        else:
            raise NotImplementedError(f'Task {TASK} is not supported for DPO.')
        val_data = val_data.map(dpo_one_pair_format, batched=True, remove_columns=val_data.column_names)

        data_module = dict(
            train_dataset=train_data, 
            eval_dataset=val_data,
            EVAL_MAX_LEN=EVAL_MAX_LEN,
            EVAL_BATCH_SIZE=EVAL_BATCH_SIZE,
            SAMPLE=SAMPLE,
            TEMP=TEMP,
            TRAIN_DATA_PATH=TRAIN_DATA_PATH,
        )

        train_args.prediction_loss_only = True
        train_args.eval_strategy = 'steps'
        train_args.logging_strategy = 'steps'
        train_args.logging_steps = 100

        train_args.metric_for_best_model = 'pass@1'
        train_args.greater_is_better = True
        # train_args.metric_for_best_model = 'eval_loss'
        # train_args.greater_is_better = False

        args_as_dict = train_args.to_dict()
        args_as_dict['generate_during_eval'] = True
        args_as_dict['max_length'] = MAX_LEN
        train_args = DPOConfig(**args_as_dict)
        trainer = MyDPOTrainer(
            model=model, 
            processing_class=tokenizer, 
            args=train_args, 
            **data_module, 
            ref_model=model.name_or_path, 
        )

    elif OPTIM == 'dpo_dyn':
        # bypass the original tokenizer to support dynamic sampling
        train_data = train_data.map(psudo_format, fn_kwargs={'column_name': 'prompt'})
        train_data = train_data.map(psudo_format, fn_kwargs={'column_name': 'chosen'})
        train_data = train_data.map(psudo_format, fn_kwargs={'column_name': 'rejected'})
        val_data = val_data.map(psudo_format, fn_kwargs={'column_name': 'prompt'})
        val_data = val_data.map(psudo_format, fn_kwargs={'column_name': 'chosen'})
        val_data = val_data.map(psudo_format, fn_kwargs={'column_name': 'rejected'})

        val_data = val_data.map(dpo_dyn_format, remove_columns=['answers'], fn_kwargs={'task': 'pvf'})

        data_collator = MyDPOCollator(
            tokenizer=tokenizer, 
            pad_token_id=tokenizer.pad_token_id,
        )
        data_collator.task = TASK

        data_module = dict(
            data_collator=data_collator,
            train_dataset=train_data, 
            eval_dataset=val_data,
            EVAL_MAX_LEN=EVAL_MAX_LEN,
            EVAL_BATCH_SIZE=EVAL_BATCH_SIZE,
            SAMPLE=SAMPLE,
            TEMP=TEMP,
            TRAIN_DATA_PATH=TRAIN_DATA_PATH,
        )

        train_args.prediction_loss_only = True

        train_args.metric_for_best_model = 'pass@1'
        train_args.greater_is_better = True
        # train_args.metric_for_best_model = 'eval_loss'
        # train_args.greater_is_better = False

        args_as_dict = train_args.to_dict()
        args_as_dict['generate_during_eval'] = True
        args_as_dict['max_length'] = MAX_LEN
        train_args = DPOConfig(**args_as_dict)
        trainer = MyDPOTrainer(
            model=model, 
            processing_class=tokenizer, 
            args=train_args, 
            **data_module, 
            ref_model=model.name_or_path, 
        )
    else:
        raise NotImplementedError

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        trainer.train()

    print('Train done!')


if __name__ == '__main__':
    main()
    time.sleep(60)
