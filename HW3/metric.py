import os
import numpy as np
import torch
torch.cuda.is_available()

from tw_rouge import get_rouge
from typing import List, Dict
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


## from eval.py
def rouge_score(predictions, references, avg=True):
    predictions = [pred.strip() + '\n' for pred in predictions]
    references = [ref.strip() + '\n' for ref in references]
    return get_rouge(predictions, references, avg)

def reward_batch(pred: List, ref: List, weight: List):
    score = rouge_score(pred, ref, avg=False)
    baseline = weight[0]*0.225 + weight[1]*0.093 + weight[2]*0.21
    reward = [weight[0]*item['rouge-1']['f'] + \
                weight[1]*item['rouge-2']['f'] + \
                 weight[2]*item['rouge-l']['f'] for item in score]
    return torch.FloatTensor(reward) - baseline
def pg_loss(logits, labels, pred, ref, weight):
    device = logits.device
    reward = reward_batch(pred, ref, weight).to(device)
    # print(reward.shape)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    ce_loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(logits.shape[0], -1).mean(dim=1)
    
    rl_loss = (ce_loss * reward).sum()
    del reward
    return rl_loss
    