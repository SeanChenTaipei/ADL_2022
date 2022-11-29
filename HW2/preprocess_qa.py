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
    # parser.add_argument(
    #     "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
    # )
    parser.add_argument(
        "--context_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument("-d", "--output_dir", default="./", type=Path)
    args = parser.parse_args()    
    return args

args = parse_args()
train_data = json.loads(Path(args.train_file).read_text())
valid_data = json.loads(Path(args.validation_file).read_text())
# test_data = json.loads(Path(args.test_file).read_text())
context = json.loads(Path(args.context_file).read_text()) ## List

def encoder(data):
    swag = {
        'id': data['id'],
        'title': data['id'],
        'context': context[data['relevant']],
        'question': data['question'],
        'answers': {'text': [data['answer']['text']], 'answer_start': [data['answer']['start']]}
    }
    return swag
def encoder2(data):
    swag = {
        'id': data['id'],
        'title': data['id'],
        'context': context[data['relevant']],
        'question': data['question']}
    return swag

new_train = list(map(encoder, train_data))
new_valid = list(map(encoder, valid_data))
# new_test = list(map(encoder2, test_data))

args.output_dir.mkdir(parents=True, exist_ok=True)
with open(args.output_dir / 'train_qa.json', 'w+') as f:
    json.dump(new_train, f)
with open(args.output_dir / 'valid_qa.json', 'w+') as f:
    json.dump(new_valid, f)
# with open(args.output_dir / 'test_qa.json', 'w+') as f:
#     json.dump(new_test, f)
