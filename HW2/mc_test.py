import argparse
import json
import logging
import math
import os
import random
import numpy as np
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, get_full_repo_name, send_example_telemetry



logger = get_logger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--out_file", type=str, default="./mc_pred.json")
    args = parser.parse_args()
    
    return args

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # label_name = "label" if "label" in features[0].keys() else "labels"
        # labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        # batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    config = AutoConfig.from_pretrained(args.ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(args.ckpt_dir, config=config)
    # data_files["train"] = Dataset.from_list(json.loads(Path('./mc_data/train_mc.json').read_text()))
    data_files = {}
    data_files["test"] = Dataset.from_list(json.loads(Path(args.test_file).read_text()))
    raw_datasets = datasets.DatasetDict(data_files)
    column_names = raw_datasets['test'].column_names

    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"

    test_examples = raw_datasets["test"]
    #test_examples = test_examples.select(range(10))
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = False
    def preprocess_test_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
        # labels = examples[label_column_name]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=512,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs["labels"] = labels
        return tokenized_inputs
    with accelerator.main_process_first():
        processed_datasets = test_examples.map(
            preprocess_test_function, batched=True, remove_columns=column_names
        )
    test_dataset = processed_datasets
    data_collator = DataCollatorForMultipleChoice(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.test_batch_size)

    # Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    # print(model.device)

    # Test!
    logger.info("\n******** Running predicting ********")
    logger.info(f"Num test examples = {len(test_dataset)}")

    test_dataset.set_format(columns=["attention_mask", "input_ids", "token_type_ids"])
    model.eval()
    all_logits = []
    for step, data in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            outputs = model(**data)
            all_logits.append(accelerator.gather(outputs.logits.argmax(dim=-1)).cpu().numpy())
        # print(all_logits)
    outputs_numpy = np.concatenate(all_logits, axis=0)
    dset = raw_datasets["test"]
    test_qa = []
    for idx, index in enumerate(outputs_numpy):
        data = {
                'id': dset[idx]['id'],
                'context': dset[idx][f'ending{index}'],
                'question': dset[idx]['sent1'],
                'answers': {'text': [dset[idx][f'ending{index}'][0]], 'answer_start': [0]}
               }
        test_qa.append(data)
    with open(args.out_file, "w+") as f:
        json.dump(test_qa, f)