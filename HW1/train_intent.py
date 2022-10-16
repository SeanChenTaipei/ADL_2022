import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from sklearn.metrics import accuracy_score

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
Best = 10
PATIENT = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# optimizer_opts = {"lr": 0.001}


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, lr_decay, last_epoch=-1): 
    if num_warmup_steps == 0:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    def lr_lambda(current_step: int):
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return lr_decay**(current_step - num_warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    
def main(args):
    global Best, PATIENT
    # print(type(args.cache_dir), args.cache_dir)
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS} ## 2 items, a dict
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    ### After model selection, add eval data into training to achieve best result.
    if args.use_all:
        train = json.loads(Path("./data/intent/train.json").read_text())
        evald = json.loads(Path("./data/intent/eval.json").read_text())
        datasets['train'] = SeqClsDataset(train+evald, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for train / dev datasets
    dataLength = len(datasets['train'])
    # print(dataLength)
    trainloader = DataLoader(datasets['train'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=datasets['train'].collate_fn,) 
    devloader = DataLoader(datasets['eval'],
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=False,
                            pin_memory=True,
                            collate_fn=datasets['eval'].collate_fn,)
    
    # loss function
    criterion = nn.CrossEntropyLoss()

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings = embeddings,
                      hidden_size = args.hidden_size,
                      num_layers = args.num_layers,
                      dropout = args.dropout,
                      bidirectional = args.bidirectional,
                      num_class = 150)
    if args.ckpt != "":
        print(f"* Loading Weight from : {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    # model = torch.nn.DataParallel(model) 
    model.to(device) ##push model to device
    model.train() ## training mode

    # TODO: init optimizer
    optimizer_func = torch.optim.AdamW
    # optimizer_func = torch.optim.RMSprop
    optimizer = optimizer_func(model.parameters(), lr = args.lr, weight_decay=1e-4) 
    # lr scheduler
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warm_up_step, args.lr_decay)
    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(-args.warm_up_step, args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        pbar = tqdm(total=int(len(datasets['train'])/args.batch_size), ascii='░▒█', ncols=110)
        if epoch < 0:
            _ = pbar.set_description(f"Warm {epoch:2d}")
        else:
            _ = pbar.set_description(f"Epoch{epoch:2d}")
        lr= scheduler.get_last_lr()[-1]
        totalLoss = 0
        trainAcc = 0
        for index, item in enumerate(trainloader):
            input_batch_i, target_batch_i = item['text'].to(device), item['intent'].to(device)
            output = model(input_batch_i)
            loss = criterion(output, target_batch_i)
            totalLoss+=loss.item()*len(target_batch_i)
            
            ## BackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            
            ## accuracy
            size = target_batch_i.shape[0]
            train_acc = accuracy_score(output.max(dim=1)[1].cpu().detach().numpy(),
                                              target_batch_i.cpu().detach().numpy())
            # trainAcc+=train_acc*(size/len(datasets['train']))
            trainAcc += (output.max(dim=1)[1].cpu().detach().numpy()==target_batch_i.cpu().detach().numpy()).sum()
            
            ## pbar update
            pbar.set_postfix(loss=loss.item(), train_acc=train_acc, lr=lr)
            pbar.update(1)
        totalLoss = totalLoss/dataLength
        trainAcc = trainAcc/len(datasets['train'])
        pbar.set_postfix(Loss=totalLoss, trainAcc=trainAcc, lr=lr)
        if epoch%5 == 0 and epoch >= 0:
            acc = 0
            with torch.no_grad():
                for item in devloader:
                    input_batch_i, target_batch_i = item['text'].to(device), item['intent']
                    output = model(input_batch_i)
                    accuracy = accuracy_score(output.max(dim=1)[1].cpu().detach().numpy(),
                                              target_batch_i.cpu().detach().numpy())
                    size = target_batch_i.shape[0]
                    acc+=accuracy*(size/len(datasets['eval']))
            pbar.set_postfix(loss=totalLoss, trainAcc=trainAcc, valAcc = acc, lr=lr)
        # TODO: Evaluation loop - calculate accuracy and save model weights
            checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': totalLoss
                         } 
            checkpoint_fname = f'ckpt_{epoch:03}_loss_{totalLoss:.2f}_valacc_{acc:.2f}.tar'
            torch.save(checkpoint, args.ckpt_dir / checkpoint_fname)
        if epoch >= 0:
            if loss.item()< Best:
                Best = loss.item()
            else:
                PATIENT += 1
            if PATIENT>=5:
                scheduler.step()
                PATIENT = 0
        else:
            scheduler.step()
        pbar.close()

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Pretrain weight path.",
        default="",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", type=bool, default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # lr decay
    parser.add_argument("--lr_decay", type=float, default=0.5)
    
    # warm up
    parser.add_argument("--warm_up_step", type=int, default=0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Use all
    # data loader
    parser.add_argument("--use_all", type=bool, default=False)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=31)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
