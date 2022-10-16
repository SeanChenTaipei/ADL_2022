from typing import List, Dict
import torch

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        test: bool = False
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.test = test

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        sample_text = [item['text'].split(" ") for item in samples]
        sample_text = [[word.replace(",", "") for word in sentence] for sentence in sample_text]
        # print(sample_text)
        sample_text = self.vocab.encode_batch(sample_text)
        if not self.test:
            label = [self.label2idx(item['intent']) for item in samples]
            return {'text': torch.LongTensor(sample_text), 'intent': torch.LongTensor(label)}
        else:
            return {'text': torch.LongTensor(sample_text)}
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(Dataset):
    def __init__(
        self, data: List[Dict], vocab: Vocab, label_mapping: Dict[str, int], max_len: int, test: bool = False):
        self.data = data
        self.vocab = vocab # 4117 words
        self.label_mapping = label_mapping ## 10 classes
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.test = test
        # ignore_idx = -100
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        instance['text_lens'] = len(instance['tokens'])
        return instance
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    def collate_fn(self, samples: List[Dict]) -> Dict:
        sample_text = [item['tokens'] for item in samples]
        sample_text = self.vocab.encode_batch(sample_text)
        
        text_lens = [item['text_lens'] for item in samples]
        if not self.test:
            target = [list(map(self.label2idx, item['tags'])) for item in samples]
            target = pad_to_len(target, len(sample_text[0]), self.label_mapping["PAD"])
            return torch.LongTensor(sample_text), torch.LongTensor(target), text_lens
        else:
            return torch.LongTensor(sample_text), text_lens
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
    
    def list2label(self, idx: List[int]):
        return list(map(self.idx2label, idx))
    
    def pred2label(self, pred: torch.Tensor, text_lens: List):
        pred = pred.max(dim=2)[1].tolist()
        tags = [self.list2label(item[:text_lens[idx]]) for idx, item in enumerate(pred)]
        return tags
        
