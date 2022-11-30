import argparse
import json
import logging
import math
import os
import random
import re
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
# from filelock import FileLock
# from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

torch.cuda.is_available()
from metric import rouge_score
print("Cuda Available ? ", torch.cuda.is_available())

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")

    parser.add_argument(
        "--prediction_file", type=str, default=None, help="A csv or a json file containing the prediction data."
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Path to store the generation."
    )
    parser.add_argument(
        "--mode", type=str, default="predict", help="predict or score"
    )
    
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default = 4)
    
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator = Accelerator()
    
    ## Dataset
    data_files = {}
    data_files["test"] = args.prediction_file
    raw_datasets = load_dataset('json', data_files=data_files)
    
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        
        
    column_names = raw_datasets["test"].column_names
    text_column = args.text_column
    
    data_id = raw_datasets["test"]["id"]
    
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    
    def parsing_text(input_txt):
        # input_txt = input_txt.lower()
        # input_txt = re.sub('\n', '', input_txt)
        # input_txt = re.sub(r'http\S+', '', input_txt)
        return input_txt
    def preprocess_function(examples):
        inputs = examples[text_column]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    pred_dataset = processed_datasets["test"]
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds
    pred_dataloader = DataLoader(pred_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    model, pred_dataloader = accelerator.prepare(model, pred_dataloader)
    model.to(device)
    
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample, 
        "top_p": args.top_p, 
        "top_k": args.top_k,
        "temperature": args.temperature,
    }
    
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(pred_dataloader)):
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions+=decoded_preds
    if args.mode == "score":
        references = raw_datasets['test'][args.summary_column]
        result = rouge_score(predictions=predictions, references=references)
        print(f"Config : {gen_kwargs}")
        print(f"rouge-1 : {round(result['rouge-1']['f']*100, 2)}")
        print(f"rouge-2 : {round(result['rouge-2']['f']*100, 2)}")
        print(f"rouge-l : {round(result['rouge-l']['f']*100, 2)}")
    
    output = zip(data_id, predictions)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as file:
        for idx, pred in output:
            file.write(json.dumps({'title': pred, 'id': idx}, ensure_ascii=False))
            file.write('\n')
        
    
    