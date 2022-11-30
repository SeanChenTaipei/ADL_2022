import os
import numpy as np
import torch
torch.cuda.is_available()

from tw_rouge import get_rouge
from typing import List, Dict
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def rouge_score(predictions, references, avg=True):
    predictions = [pred.strip() + '\n' for pred in predictions]
    references = [ref.strip() + '\n' for ref in references]
    return get_rouge(predictions, references, avg)


def rouge_score_f(predictions, references, avg=True):
    predictions = [pred.strip() + '\n' for pred in predictions]
    references = [ref.strip() + '\n' for ref in references]
    score = get_rouge(predictions, references, avg)
    score = [[item['rouge-1']['f'], item['rouge-2']['f'], item['rouge-l']['f']] for item in score]
    return np.array(score)

def reward_batch(pred: List, ref: List, weight):
    score = rouge_score_f(pred, ref, avg=False)
    baseline =  np.sum(weight * np.array([0.22, 0.085, 0.2]))
    # baseline = weight[0]*0.22 + weight[1]*0.085 + weight[2]*0.2
    reward = np.sum(weight * score, axis=1)
    # reward = [weight[0]*item['rouge-1']['f'] + \
    #             weight[1]*item['rouge-2']['f'] + \
    #              weight[2]*item['rouge-l']['f'] for item in score]
    return torch.FloatTensor(reward)/baseline
def pg_loss(logits, labels, pred, ref, weight):
    device = logits.device
    reward = reward_batch(pred, ref, weight).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    ce_loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1)).view(logits.shape[0], -1).mean(dim=1)
    
    rl_loss = (ce_loss * reward).mean() + 0.5*ce_loss.mean()
    del reward
    return rl_loss
    