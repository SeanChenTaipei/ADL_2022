from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding

from collections import OrderedDict



class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,) -> None:
        super(SeqClassifier, self).__init__()
        self.random_seed = 2
        torch.manual_seed(self.random_seed)
        
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        self.lstm = nn.LSTM(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.gru = nn.GRU(input_size=embeddings.shape[1],
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          # proj_size=hidden_size/2,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size*2, num_class)
        else:
            self.fc1 = nn.Linear(hidden_size, num_class)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)
        self.classifier = nn.Sequential(OrderedDict([
                                       ('dense1', self.fc1),
                                       ('relu1', nn.ELU(0.1)),
                                       ('dropout1', nn.Dropout(p=0.2)),
                                       ('dense2', self.fc2),
                                       ('relu2', nn.ELU(0.1)),
                                       ('dropout2', nn.Dropout(p=0.2)),
                                       ('dense3', self.fc3)]))
        self.bidirectional = bidirectional
        # TODO: model architecture
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return hidden_size*2
        else:
            return hidden_size
        # raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        
        x = self.embed(batch)
        x, _status = self.gru(x)
        if self.bidirectional:
            x = x.max(dim=1)[0]
        else:
            x = x[:, -1, :].squeeze()
        x = self.fc1(x)
        # x = self.classifier(x)
        return x


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,) -> None:
        super(SeqTagger, self).__init__()
        self.random_seed = 666
        torch.manual_seed(self.random_seed)
        
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        self.lstm = nn.LSTM(input_size=embeddings.shape[1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.gru = nn.GRU(input_size=embeddings.shape[1],
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=bidirectional)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size*2, 128)
        else:
            self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(self.fc1,
                                        nn.Hardswish(),
                                        nn.Dropout(0.3),
                                        nn.Linear(128, 128),
                                        nn.Hardswish(),
                                        nn.Dropout(0.2),
                                        nn.Linear(128, num_class))
                                        
        self.bidirectional = bidirectional
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x = self.embed(batch)
        x, _status = self.lstm(x) ## x is of shape (N, L, hidden_size*2 or hidden_size)
        # x = self.drop(x)
        x = self.classifier(x)
        return x