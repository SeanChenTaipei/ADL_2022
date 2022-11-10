import os
import sys
import json
import datasets
from dataclasses import dataclass, field
from datasets import load_dataset
from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument("-d", "--output_dir", default="./mc_data", type=Path)
    args = parser.parse_args()    
    return args

args = parse_args()


context = json.loads(Path(args.context_file).read_text()) ## List


def encoder(data):
    swag = {
        'id': data['id'],
        'sent1': data['question'],
        'sent2': '',
        **{f"ending{i}": context[data['paragraphs'][i]] for i in range(4)},
        'label': data['paragraphs'].index(data['relevant'])
    }
    return swag
def encoder_test(data):
    swag = {
        'id': data['id'],
        'sent1': data['question'],
        'sent2': '',
        **{f"ending{i}": context[data['paragraphs'][i]] for i in range(4)},
        'index': data['paragraphs']
        # 'label': data['paragraphs'].index(data['relevant'])
    }
    return swag


if args.train_file is not None:
    train_data = json.loads(Path(args.train_file).read_text())
    new_train = list(map(encoder, train_data))
if args.validation_file is not None:
    valid_data = json.loads(Path(args.validation_file).read_text())
    new_valid = list(map(encoder, valid_data))
if args.test_file is not None:
    test_data = json.loads(Path(args.test_file).read_text())
    new_test = list(map(encoder_test, test_data))




args.output_dir.mkdir(parents=True, exist_ok=True)
if args.train_file is not None:
    with open(args.output_dir / 'train_mc.json', 'w+') as f:
        json.dump(new_train, f)
if args.validation_file is not None:
    with open(args.output_dir / 'valid_mc.json', 'w+') as f:
        json.dump(new_valid, f)
if args.test_file is not None:
    with open(args.output_dir / 'test_mc.json', 'w+') as f:
        json.dump(new_test, f)
